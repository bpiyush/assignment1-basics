import os
import sys
import numpy as np
from tqdm import tqdm
from glob import glob
import json
from collections import Counter
from copy import deepcopy


def count_bytes(indices: dict):
    count = 0
    for k, v in indices.items():
        count += len(k) * v
    return count


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data


def _encode(string: str):
    return list(string.encode("utf-8"))


def pprint_list(x):
    print("\n".join(str(item) for item in x))


def show_topk_dict_entries(d, k=20):
    pprint_list(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])


def merge_optimized(indices, max_pair, new_ind, pair_counts):
    _indices = Counter()
    for k, v in indices.items():

        result = []
        i = 0
        while i < len(k):
            if i < len(k) - 1 and k[i] == max_pair[0] and k[i+1] == max_pair[1]:

                # [..... k[i - 1], **k[i], k[i+1],** k[i+2], ...]
                # 1. Replace **k[i], k[i+1]** with new_ind
                # 2. Handle left counter
                # 3. Handle right counter

                # Replacement
                result.append(new_ind)

                # Left
                if i - 1 >= 0:
                    # Decrement [k[i - 1], k[i]] by v
                    # assert (k[i - 1], k[i]) in pair_counts
                    # pair_counts[ tuple([k[i - 1], k[i]]) ] -= v
                    pair_counts[ (k[i - 1], k[i]) ] -= v

                    # Increment [k[i - 1], new_ind] by v
                    # pair_counts[ tuple([k[i - 1], new_ind]) ] += v
                    pair_counts[ (k[i - 1], new_ind) ] += v

                # Right
                if i + 2 < len(k):
                    
                    # Decrement [k[i + 1], k[i + 2]] by v
                    # assert (k[i + 1], k[i + 2]) in pair_counts
                    # pair_counts[ tuple([k[i + 1], k[i + 2]]) ] -= v
                    pair_counts[ (k[i + 1], k[i + 2]) ] -= v

                    # Increment [new_ind, k[i + 2]]
                    # pair_counts[ tuple([new_ind, k[i + 2]]) ] += v
                    pair_counts[ (new_ind, k[i + 2]) ] += v
                
                i += 2
            else:
                result.append(k[i])
                i += 1
        
        _indices[tuple(result)] += v

    return _indices, pair_counts


def train_bpe_optimized(
    input_path: str,
    vocab_size: int,
    special_tokens: list=['<|endoftext|>'],
):

    # Load the pre-tokenized counts
    pretok_counts = load_json(input_path)

    # Convert strings into list of bytes
    tok_seq_counts = {
        tuple(_encode(k)): v for k, v in pretok_counts.items()
    }

    # Initialize vocabulary
    vocab = {k: bytes([k]) for k in range(256)}

    # Initialize merges
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = {}

    # Make a copy of the token seq. counts dict.
    indices = deepcopy(tok_seq_counts)

    # Maintain a single dict to count the frequency of byte pairs
    # At the start, fill it with initial counts: [a, b] = count("a, b")
    # where a/b are in [0, 255].
    pair_counts = Counter()
    for k, v in indices.items():
        if len(k) < 2:
            # Nothing to add
            continue
        for j in range(len(k) - 1):
            pair_counts[tuple([k[j], k[j+1]])] += v


    # Training loop
    for i in tqdm(range(num_merges), desc="Training BPE", bar_format='{l_bar}{bar:20}{r_bar}'):

        new_ind = 256 + i

        # Find the byte pair with max. frequency
        max_pair = max(pair_counts, key=pair_counts.get)
        del pair_counts[max_pair]

        # Update the vocab and merges
        merges[max_pair] = new_ind
        vocab[new_ind] = vocab[max_pair[0]] + vocab[max_pair[1]]

        # Merge + update the count dict
        indices, pair_counts = merge_optimized(indices, max_pair, new_ind, pair_counts)
        # import ipdb; ipdb.set_trace()
        # {k: v for k, v in indices.items() if max_pair in zip(k, k[1:])}
        # Only for debugging: checks if a particular triplet count matches
        # assert sum([v for k, v in indices.items() if (max_pair[0], max_pair[1], 111) in zip(k, k[1:], k[2:])]) == pair_counts[(new_ind, 111)]
        

    # Add special tokens to the vocabulary
    n = len(vocab)
    for j, s in enumerate(special_tokens):
        vocab[n+j] = s.encode("utf-8")

    # TODO: fix compute compression ratio
    # NOTE: since we have removed special tokens, those do not contribute to the n.o. bytes
    comp_ratio = count_bytes(tok_seq_counts) / count_bytes(indices)
    print("Trained BPE with compression ratio (approx.): ", comp_ratio)

    return vocab, merges


if __name__ == "__main__":

    # Load pre-tokenization counts
    data_dir = "/scratch/shared/beegfs/piyush/datasets/text_data"
    mode = "train"
    pretok_filepath = f"{data_dir}/TinyStoriesV2-GPT4-{mode}-pretokenization_counts.json"
    text_filepath = f"{data_dir}/TinyStoriesV2-GPT4-{mode}.txt"

    # Test it out
    vocab, merges = train_bpe_optimized(pretok_filepath, vocab_size=256+1+10000, special_tokens=['<|endoftext|>'])
    
    # Sanity check
    i = np.argmax([len(vocab[i]) for i in vocab])
    print("Vocab entry with max. length: ", i, vocab[i])

    import regex as re
    from cs336_basics.pretokenization import find_chunk_boundaries

    query = (vocab[i]).decode("utf-8")
    with open(text_filepath, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        c = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            match_iter = re.finditer(re.compile(query), chunk)
            for m in match_iter:
                c += 1
            # del chunk
    print(f"Number of times `{query}` appears in the corpus: ", c)
    
    # Save output
    output = dict(vocab=vocab, merges=merges)
    np.savez(f"{data_dir}/TinyStoriesV2-GPT4-{mode}-bpe_optimized.npz", **output)
    print("Saved output to: ", f"{data_dir}/TinyStoriesV2-GPT4-{mode}-bpe_optimized.npz")
