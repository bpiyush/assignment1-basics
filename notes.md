
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



### Steps

Input: A large text file with <|endoftext|> as the special token that separates documents. It is present at the start of each document.

Process.

1. First, using multi-processing, splitting over the special tokens, use 1 CPU to process 1 chunk. Each chunk is a subset of the input text.
2. Pre-tokenization: In each document, remove all the special tokens. Then, pre-tokenize each document with regex. Then, create a dict where
   keys are pre-tokens and values are their frequency in the corpus. Note that this is parallelizable across chunks. (`pretokenization.py`)
3. BPE: Initialize the vocabulary and merges dicts. Run over the pre-token frequency dict and get the byte pair with highest frequency. Then,
   merge this pair in the pre-token frequency dict. Maintain a dict for pair counts. (`bpe_optim.py`)
   - Once you pick the max-frequency pair, remove it from the pair counts dict.
   - Next, update the pre-token frequency dict + the pair counts dict.
        - If the pair does not exist in a pre-token, there will be no update to the pair counts dict / pre-token frequency dict.
        - If the pair exists, then merge it in the key. Plus, update pair counts affected by this merge (left and right). Also, update the 
        vocab and merges dicts.
4. Inference: once the BPE is trained, given the vocab and merges dicts, we need to use it to encode/decode. (`tokenizer.py`)
   - Encode: First, split across chunks taking care of special tokens. Each chunk needs to be processed separately. For each chunk, first,
     manage the special tokens separately. Then, for each document (separated by special tokens), pre-tokenize. Apply merges to each
     pre-token independently.
   - Decode: given a sequence of integers, convert each integer to its byte sequence, concatenate it and decode.

Output: A .pt file mapping the entire corpus into a large sequence of integers [0-vocab_size].


* OpenWebText (OWT): Train set has 2,727,181,568 (2.7B) tokens and validation set has 66,399,684 (66M) tokens.
* TinyStories (TS): Train set has 541,124,931 (0.5B) tokens and validation set has 5,464,803 (5M) tokens.