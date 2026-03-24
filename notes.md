
On `OpenWebText`:

```txt
Loading pre-tokenization counts from:  /scratch/shared/beegfs/piyush/datasets/text_data/owt_train-pretokenization_counts.json
Training BPE: 100%|████████████████████| 32000/32000 [3:27:28<00:00,  2.57it/s]
Trained BPE with compression ratio (approx.):  4.36340160492044
Vocab entry with max. length:  31287 b'---------------------------'
Number of times `---------------------------` appears in the corpus:  29397
Saved output to:  /scratch/shared/beegfs/piyush/datasets/text_data/owt_train-bpe_optimized.npz
```

Note that pre-tokenization takes about 2:13 hours across 16 CPU cores.

On `TinyStoriesV2-GPT4`:

```txt
Loading pre-tokenization counts from:  /scratch/shared/beegfs/piyush/datasets/text_data/TinyStoriesV2-GPT4-train-pretokenization_counts.json
Training BPE: 100%|████████████████████| 10000/10000 [00:29<00:00, 340.68it/s]
Trained BPE with compression ratio (approx.):  4.072053539949478
Vocab entry with max. length:  7167 b' accomplishment'
Number of times ` accomplishment` appears in the corpus:  1516
Saved output to:  /scratch/shared/beegfs/piyush/datasets/text_data/TinyStoriesV2-GPT4-train-bpe_optimized.npz
```

Note that pre-tokenization takes about 29 seconds across 16 CPU cores.