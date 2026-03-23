import os
import sys
import numpy as np
from tqdm import tqdm
from glob import glob
import json
from collections import Counter
from copy import deepcopy


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


def count_byte_pairs(token_sequence_counts):
    """Returns the max. frequency byte pair."""
    pair_counts = Counter()
    for k, v in token_sequence_counts.items():
        if len(k) < 2:
            # Nothing to add
            continue
        k = list(k)
        for j in range(len(k) - 1):
            pair_counts[tuple([k[j], k[j+1]])] += v
    max_pair = max(pair_counts, key=pair_counts.get)
    return max_pair


def _merge(k, pair, n):
    result = []
    i = 0
    while i < len(k):
        if i < len(k) - 1 and k[i] == pair[0] and k[i+1] == pair[1]:
            result.append(n)
            i += 2
        else:
            result.append(k[i])
            i += 1
    return result


def merge(ids: dict, pair: tuple, n: int):
    _ids = Counter()
    for k, v in ids.items():
        k = list(k)
        _k = _merge(k, pair, n)
        _ids[tuple(_k)] += v
    return _ids


def count_bytes(indices: dict):
    count = 0
    for k, v in indices.items():
        count += len(k) * v
    return count


def train_bpe(
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

    # Training loop
    for i in tqdm(range(num_merges), desc="Training BPE", bar_format='{l_bar}{bar:20}{r_bar}'):

        # 1. Count the frequency of byte pairs & find max. pair
        max_pair = count_byte_pairs(indices)

        # 2. Merge the max_pair entries in the indices
        new_ind = 256 + i
        indices = merge(indices, max_pair, new_ind)
        merges[max_pair] = new_ind
        vocab[new_ind] = vocab[max_pair[0]] + vocab[max_pair[1]]

    # Add special tokens to the vocabulary
    for s in special_tokens:
        vocab[len(vocab)] = s

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
    vocab, merges = train_bpe(pretok_filepath, vocab_size=256+1+10000, special_tokens=['<|endoftext|>'])
    
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
