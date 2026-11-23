import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "roneneldan/TinyStories"
VOCAB_SIZE = 8192
TOKENIZER_SAMPLE_BYTES = 10 * 1024 * 1024  # 10 MB
SEQ_LEN = 256
WINDOW_LEN = SEQ_LEN + 1  # 257 tokens for 256 next-token predictions
MIN_TRAIN_TARGET_TOKENS = 33_554_432
NUM_MONITOR_WINDOWS = 1024
NUM_AUDIT_WINDOWS = 4096
SEED = 42


def normalize_text(text: str) -> str:
    """Normalize line endings and collapse consecutive whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_data_dir = Path("data/processed/tinystories")
    out_manifest_dir = Path("manifests/tinystories")
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pin dataset revision
    api = HfApi()
    ds_info = api.dataset_info(DATASET_ID)
    pinned_revision = ds_info.sha
    print(f"Loading {DATASET_ID} at pinned revision: {pinned_revision}")

    ds_train = load_dataset(DATASET_ID, split="train",
                            revision=pinned_revision)
    ds_val = load_dataset(DATASET_ID, split="validation",
                          revision=pinned_revision)

    # 2. Normalize, hash, and deduplicate across and within splits
    seen_hashes = set()
    dups_removed = {"train_intra": 0, "val_intra": 0, "cross_split": 0}

    train_docs = []
    for idx, row in enumerate(ds_train):
        norm = normalize_text(row["text"])
        if not norm:
            continue
        doc_hash = sha256_str(norm)
        if doc_hash in seen_hashes:
            dups_removed["train_intra"] += 1
            continue
        seen_hashes.add(doc_hash)
        train_docs.append({"doc_id": f"train_{idx:07d}",
                          "hash": doc_hash, "text": norm})

    train_hash_set = {d["hash"] for d in train_docs}
    monitor_docs = []
    audit_docs = []

    for idx, row in enumerate(ds_val):
        norm = normalize_text(row["text"])
        if not norm:
            continue
        doc_hash = sha256_str(norm)
        if doc_hash in train_hash_set:
            dups_removed["cross_split"] += 1
            continue
        if doc_hash in seen_hashes:
            dups_removed["val_intra"] += 1
            continue
        seen_hashes.add(doc_hash)

        # Deterministic 50/50 split of validation into monitor vs audit via hash parity
        bucket = int(doc_hash[:8], 16) % 2
        record = {"doc_id": f"val_{idx:07d}", "hash": doc_hash, "text": norm}
        if bucket == 0:
            monitor_docs.append(record)
        else:
            audit_docs.append(record)

    # Sort deterministically by hash so order is invariant to loader shuffling
    train_docs.sort(key=lambda d: d["hash"])
    monitor_docs.sort(key=lambda d: d["hash"])
    audit_docs.sort(key=lambda d: d["hash"])

    print(
        f"Unique docs -> Train: {len(train_docs)}, Monitor: {len(monitor_docs)}, Audit: {len(audit_docs)}")
    print(f"Duplicates removed: {dups_removed}")

    # 3. Build 10 MB tokenizer training sample strictly from train_docs
    held_out_hashes = {d["hash"]
                       for d in monitor_docs} | {d["hash"] for d in audit_docs}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_docs))

    sample_path = out_data_dir / "tokenizer_train_sample.txt"
    tokenizer_sample_doc_ids = []
    bytes_written = 0

    with open(sample_path, "w", encoding="utf-8") as f:
        for i in perm:
            doc = train_docs[int(i)]
            assert doc["hash"] not in held_out_hashes, "LEAKAGE DETECTED: Held-out hash in tokenizer sample!"
            encoded = (doc["text"] + "\n").encode("utf-8")
            f.write(doc["text"] + "\n")
            tokenizer_sample_doc_ids.append(doc["doc_id"])
            bytes_written += len(encoded)
            if bytes_written >= TOKENIZER_SAMPLE_BYTES:
                break

    sample_sha256 = sha256_file(sample_path)
    print(
        f"Wrote {bytes_written:,} bytes ({len(tokenizer_sample_doc_ids)} train docs) to {sample_path}")

    # 4. Fit 8,192-token SentencePiece BPE tokenizer with byte fallback
    spm_prefix = str(out_manifest_dir / "spm_tinystories_8192")
    spm_args = {
        "input": str(sample_path),
        "model_prefix": spm_prefix,
        "vocab_size": VOCAB_SIZE,
        "model_type": "bpe",
        "character_coverage": 1.0,
        "byte_fallback": True,
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        "pad_piece": "<pad>",
        "unk_piece": "<unk>",
        "bos_piece": "<bos>",
        "eos_piece": "<eos>",
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "seed_sentencepiece_size": 1000000,
    }
    spm.SentencePieceTrainer.train(**spm_args)

    sp = spm.SentencePieceProcessor(model_file=f"{spm_prefix}.model")
    eos_id = sp.eos_id()
    assert eos_id == 3, f"Expected eos_id=3, got {eos_id}"
    tokenizer_sha256 = sha256_file(Path(f"{spm_prefix}.model"))

    # 5. Build fixed training token stream (>= 33,554,432 target tokens)
    # Using deterministic shuffle order for the core training stream
    train_stream_perm = np.random.default_rng(
        SEED + 1).permutation(len(train_docs))
    train_tokens = []
    train_stream_doc_ids = []

    for i in train_stream_perm:
        doc = train_docs[int(i)]
        ids = sp.encode(doc["text"], out_type=int) + [eos_id]
        train_tokens.extend(ids)
        train_stream_doc_ids.append(doc["doc_id"])
        # Need at least MIN_TRAIN_TARGET_TOKENS + 1 so shifted windows cover full horizon
        if len(train_tokens) >= MIN_TRAIN_TARGET_TOKENS + WINDOW_LEN:
            break

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_npy_path = out_data_dir / "train_tokens.npy"
    np.save(train_npy_path, train_arr)
    train_shard_sha256 = sha256_file(train_npy_path)

    # 6. Build fixed non-overlapping evaluation windows (257 tokens -> 256 shifted targets)
    def build_eval_windows(docs, max_windows):
        stream = []
        used_doc_ids = []
        for doc in docs:
            stream.extend(sp.encode(doc["text"], out_type=int) + [eos_id])
            used_doc_ids.append(doc["doc_id"])
            if len(stream) >= max_windows * WINDOW_LEN:
                break
        n_available = len(stream) // WINDOW_LEN
        n_windows = min(max_windows, n_available)
        trimmed = np.array(stream[: n_windows * WINDOW_LEN], dtype=np.uint16)
        windows = trimmed.reshape(n_windows, WINDOW_LEN)
        return windows, used_doc_ids

    monitor_windows, monitor_used_ids = build_eval_windows(
        monitor_docs, NUM_MONITOR_WINDOWS)
    audit_windows, audit_used_ids = build_eval_windows(
        audit_docs, NUM_AUDIT_WINDOWS)

    monitor_npy_path = out_data_dir / "monitor_windows.npy"
    audit_npy_path = out_data_dir / "audit_windows.npy"
    np.save(monitor_npy_path, monitor_windows)
    np.save(audit_npy_path, audit_windows)

    # 7. Write frozen dataset & split manifest
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": pinned_revision,
        "license": "CDLA-Sharing-1.0",
        "normalization_rule": "CRLF->LF, collapse horizontal whitespace, max 2 consecutive newlines, strip",
        "split_rule": "Official train for LM train; official validation split 50/50 into monitor (even sha256[:8]) and audit (odd sha256[:8])",
        "duplicates_removed": dups_removed,
        "unique_document_counts": {
            "train": len(train_docs),
            "monitor": len(monitor_docs),
            "audit": len(audit_docs),
        },
        "tokenizer": {
            "model_path": f"{spm_prefix}.model",
            "model_sha256": tokenizer_sha256,
            "sample_bytes": bytes_written,
            "sample_sha256": sample_sha256,
            "sample_doc_count": len(tokenizer_sample_doc_ids),
            "settings": spm_args,
        },
        "token_streams": {
            "sequence_length": SEQ_LEN,
            "window_length_with_shift": WINDOW_LEN,
            "train_total_tokens": int(train_arr.size),
            "train_target_tokens_available": int(train_arr.size - 1),
            "train_shard_sha256": train_shard_sha256,
            "monitor_windows_shape": list(monitor_windows.shape),
            "monitor_target_tokens": int(monitor_windows.shape[0] * SEQ_LEN),
            "monitor_sha256": sha256_file(monitor_npy_path),
            "audit_windows_shape": list(audit_windows.shape),
            "audit_target_tokens": int(audit_windows.shape[0] * SEQ_LEN),
            "audit_sha256": sha256_file(audit_npy_path),
        },
    }

    manifest_path = out_manifest_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save ordered document IDs & hashes for auditability
    with open(out_manifest_dir / "split_doc_hashes.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_hashes": [d["hash"] for d in train_docs],
                "monitor_hashes": [d["hash"] for d in monitor_docs],
                "audit_hashes": [d["hash"] for d in audit_docs],
                "tokenizer_sample_doc_ids": tokenizer_sample_doc_ids,
                "train_stream_doc_ids": train_stream_doc_ids,
            },
            f,
            indent=2,
        )

    print(f"Successfully wrote frozen manifest to {manifest_path}")
    print(f"Tokenizer SHA256: {tokenizer_sha256}")
    print(f"Train Shard SHA256: {train_shard_sha256}")
    print(
        f"Monitor Windows: {monitor_windows.shape} ({monitor_windows.shape[0]*SEQ_LEN:,} target tokens)")
    print(
        f"Audit Windows: {audit_windows.shape} ({audit_windows.shape[0]*SEQ_LEN:,} target tokens)")


if __name__ == "__main__":
    main()
