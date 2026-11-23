# Week 2 TinyStories T4 pilot results

## Purpose and setup

This pilot tested how early training rankings can change by the end of a fixed training budget. Twelve recipes trained the same decoder-only language model on TinyStories in Google Colab on a Tesla T4. Each used seed 42 and ran for 2,048 updates. The recipes varied peak learning rate, warmup length, learning-rate schedule, and weight decay. Held-out monitor loss was recorded before training and every 128 updates. Lower loss means better next-token predictions on the held-out examples.

## Results

All 12 runs completed. Training losses and gradient norms stayed finite, and both the 128-update average training loss and the held-out monitor loss fell at every recorded interval. Among the recipes tested, those with a peak learning rate of 0.002 finished with monitor losses of 2.11–2.16, compared with 3.33–3.38 for recipes using 0.0001.

| Training step | Pilot 02 monitor loss (rank) | Pilot 04 monitor loss (rank) |
| ------------: | ---------------------------: | ---------------------------: |
|           512 |                2.65790 (1st) |                2.68649 (5th) |
|         1,152 |                2.28725 (1st) |                2.28820 (2nd) |
|         1,280 |                2.24839 (2nd) |                2.24838 (1st) |
|         2,048 |                2.11727 (2nd) |            **2.11126 (1st)** |

Pilot 02 and Pilot 04 used the same peak learning rate (0.002), cosine schedule, and weight decay (0.1). Pilot 02 used 2% warmup and Pilot 04 used 8%. Pilot 04 first led at a recorded checkpoint at step 1,280 and finished ahead by about 0.006 monitor loss.

## Compute and checkpoint resume

The runs processed 402.7 million training tokens in total and used 104.30 minutes of combined elapsed session time. Training took 95.54 minutes and monitor evaluations took 4.71 minutes (4.52% of elapsed time). By step 512, one quarter of the update budget, all 12 runs had used 26.40 minutes of training and evaluation time.

In a separate Pilot 01 check, an uninterrupted run and a run resumed from step 512 both reached a final monitor loss of 3.3310 to four decimal places. The uninterrupted session took 8.64 minutes and the paused-and-resumed workflow took 10.05 minutes.

## Implication for early stopping

A rule that kept only the top four recipes at step 512 would have discarded Pilot 04, the eventual winner. The saved full-length curves make it possible to replay stopping rules using only the data available at each decision point, then compare compute savings with the quality of the best recipe retained. This pilot demonstrates one ranking reversal under a single seed. More seeds are needed to estimate how often early stopping would discard a strong recipe.

[Analysis notebook](analyze_pilots.ipynb) · [Run configurations](configs/) · [Saved results](results/) · [Reproduction instructions](README.md)
