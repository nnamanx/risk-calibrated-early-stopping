import json
from pathlib import Path
import numpy as np
import sentencepiece as spm


def test_no_split_or_tokenizer_leakage():
    manifest_dir = Path("manifests/tinystories")
    data_dir = Path("data/processed/tinystories")

    with open(manifest_dir / "dataset_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(manifest_dir / "split_doc_hashes.json", "r", encoding="utf-8") as f:
        hashes = json.load(f)

    train_set = set(hashes["train_hashes"])
    monitor_set = set(hashes["monitor_hashes"])
    audit_set = set(hashes["audit_hashes"])

    # 1. Verify strictly disjoint document splits
    assert len(train_set & monitor_set) == 0, "Overlap between train and monitor!"
    assert len(train_set & audit_set) == 0, "Overlap between train and audit!"
    assert len(monitor_set & audit_set) == 0, "Overlap between monitor and audit!"

    # 2. Verify every tokenizer sample doc ID belongs exclusively to train_*
    for doc_id in hashes["tokenizer_sample_doc_ids"]:
        assert doc_id.startswith(
            "train_"), f"Non-train doc in tokenizer sample: {doc_id}"

    # 3. Verify token arrays, shapes, and vocabulary bounds
    sp = spm.SentencePieceProcessor(
        model_file=manifest["tokenizer"]["model_path"])
    assert sp.get_piece_size() == 8192

    train_tokens = np.load(data_dir / "train_tokens.npy")
    monitor_windows = np.load(data_dir / "monitor_windows.npy")
    audit_windows = np.load(data_dir / "audit_windows.npy")

    assert train_tokens.size >= 33_554_432 + 257
    assert monitor_windows.shape == (1024, 257)
    assert audit_windows.shape == (4096, 257)
    assert int(train_tokens.max()) < 8192
    assert int(monitor_windows.max()) < 8192
    assert int(audit_windows.max()) < 8192
