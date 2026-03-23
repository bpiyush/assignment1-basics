import os
from typing import BinaryIO
from tqdm import tqdm
import regex as re
from collections import Counter
from multiprocessing import Pool
import numpy as np
import json


# Pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def init_worker(num_workers=8):
    pid = os.getpid()
    # Restrict to a subset of CPUs, e.g., cores 0-3
    os.sched_setaffinity(pid, set(list(range(num_workers))))


def pretokenize_document(d: str):
    """
    Uses GPT2 regex-based pre-tokenization and returns a dict with pre token mapped to its count.
    """
    match_iter = re.finditer(PAT, d)
    counts = Counter()
    for m in match_iter:
        counts[d[m.start():m.end()]] += 1
    return counts


def pretokenize_chunk(chunk: str, special_tokens=['<|endoftext|>']):
    # Split documents around special tokens
    pattern = "|".join(re.escape(token) for token in special_tokens)
    pattern = re.compile(pattern)
    docs = re.split(pattern, chunk)

    # Run pre-tokenization on each document and gather a single count dict
    counts_chunk = Counter()
    iterator = tqdm(docs, desc="Running pre-tokenization for documents in a chunk")
    for d in iterator:
        counts_chunk += pretokenize_document(d)
    return counts_chunk


def process_chunk(args):
    filepath, start, end = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_chunk(chunk)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default="TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("-n", "--num-processes", type=int, default=8)
    args = parser.parse_args()
    data_dir = "/scratch/shared/beegfs/piyush/datasets/text_data"

    filepath = f"{data_dir}/{args.filename}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    print("Loading: ", filepath)

    # Find chunk boundaries
    with open(filepath, "rb") as f:
        num_processes = args.num_processes
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Each worker opens the file independently and reads only its chunk
    tasks = [(filepath, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_processes, initializer=init_worker, initargs=(args.num_processes,)) as pool:
        chunk_counts = pool.map(process_chunk, tasks)

    counts = Counter()
    for c in chunk_counts:
        counts += c
    
    # Inspect
    print("Number of entries in counts: ", len(counts))
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_counts[:20])

    save_name = f"{args.filename.replace('.txt', '')}-pretokenization_counts.json"
    save_path = f"{data_dir}/{save_name}"
    with open(save_path, "w") as f:
        json.dump(counts, f)
