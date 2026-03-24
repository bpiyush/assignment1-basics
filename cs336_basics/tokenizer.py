import numpy as np
from abc import ABC
from termcolor import colored
import random
import regex as re
import os
from cs336_basics.pretokenization import (
    PAT as REGEX_PATTERN,
    # find_chunk_boundaries,
    # pretokenize_chunk,
    # pretokenize_document,
)
from typing import BinaryIO
import os
import regex as re
from tqdm import tqdm


def iter_text_chunks(
    path: str,
    special_tokens: list[str],
    num_boundaries: int = 1024,
    encoding: str = "utf-8",
):
    """
    Yield chunks of text from a file without loading the whole file into memory.
    Chunk boundaries are moved forward until one of the special tokens is found.

    Parameters
    ----------
    path : str
        Path to the text file.
    special_tokens : Sequence[str]
        Special tokens to split on, e.g. ["<|endoftext|>"].
    num_boundaries : int
        Desired number of chunks.
    encoding : str
        Text encoding used to decode chunks.

    Yields
    ------
    str
        One decoded text chunk at a time.
    """
    token_bytes = [tok.encode(encoding) for tok in special_tokens]
    if not token_bytes:
        raise ValueError("special_tokens must not be empty")

    with open(path, "rb") as f:
        # File size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        if file_size == 0:
            return

        # Initial equally spaced guesses
        num_boundaries = max(1, num_boundaries)
        chunk_size = max(1, file_size // num_boundaries)
        boundaries = [i * chunk_size for i in range(num_boundaries)]
        boundaries.append(file_size)
        boundaries[-1] = file_size

        mini_chunk_size = 4096

        # Move each internal boundary forward until a special token is found
        for i in range(1, len(boundaries) - 1):
            pos = boundaries[i]
            f.seek(pos)

            while True:
                block = f.read(mini_chunk_size)
                if block == b"":
                    boundaries[i] = file_size
                    break

                found_positions = [
                    block.find(tok) for tok in token_bytes if block.find(tok) != -1
                ]

                if found_positions:
                    boundaries[i] = pos + min(found_positions)
                    break

                pos += mini_chunk_size

        # Remove duplicates and keep order
        boundaries = sorted(set(boundaries))

        # Yield chunks one at a time
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start >= end:
                continue
            f.seek(start)
            raw = f.read(end - start)
            yield raw.decode(encoding, errors="ignore")


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


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# first two helper functions...
def replace_control_characters(s: str) -> str:
    import unicodedata
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class Tokenizer(ABC):
    # Pre-tokenization pattern
    PAT = REGEX_PATTERN

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverted_vocab = {v: k for k, v in self.vocab.items()}
        self.inverted_merges = {idx: pair for pair, idx in self.merges.items()}

    @classmethod
    def from_file(cls, filepath: str, special_tokens=None):
        data = np.load(filepath, allow_pickle=True)
        vocab = data["vocab"].item()   # if it was a dict
        merges = data["merges"].item() # if it was a dict
        
        if special_tokens is not None:
            # Check if it already exists in the vocab
            for special_token in special_tokens:
                token_bytes = special_token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[len(vocab)] = token_bytes

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def peek_into_vocab(self, n: int = 10):
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        idx_to_print = np.random.choice(list(self.vocab.keys()), size=n, replace=False)
        print(colored("Showing random merged tokens from the vocabulary:", "yellow"))
        for idx in idx_to_print:
            token = self.vocab[idx]
            # note: many tokens may be partial utf-8 sequences
            # and cannot be decoded into valid strings. Here we're using
            # errors='replace' to replace them with the replacement char �.
            # this also means that we couldn't possibly use .vocab in load()
            # because decoding in this way is a lossy operation!
            s = render_token(token)
            # find the children of this token, if any
            if idx in inverted_merges:
                # if this token has children, render it nicely as a merge
                idx0, idx1 = inverted_merges[idx]
                s0 = render_token(self.vocab[idx0])
                s1 = render_token(self.vocab[idx1])
                print(f"[{s0}][{s1}] -> [{s}] {idx}")
            else:
                # otherwise this is leaf token, just print it
                # (this should just be the first 256 tokens, the bytes)
                print(f"[{s}] {idx}")

    def __repr__(self):
        self.peek_into_vocab()
        return colored(
            f"Tokenizer(vocab={len(self.vocab)}, merges={len(self.merges)}, special_tokens={self.special_tokens})",
            "green",
        )
    
    def _encode_tokens(self, tokens: list, verbose: bool = False) -> list:
        """Given a list of int (byte) IDs, return a list of int IDs after merging."""
        while len(tokens) > 1:
            stats = get_stats(tokens)
            if verbose:
                print("Stats: ", stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                if verbose:
                    print(f"pair {pair} not in merges. Leaving as is.")
                # Nothing to merge
                break
            # Merge it
            new_ind = self.merges[pair]
            if verbose:
                print("Merging: ", pair, "->", new_ind)
            tokens = merge(tokens, pair, new_ind)
        return tokens
    
    def _encode_document(self, document: str, verbose: bool = False) -> list:
        """A document is assumed to be a string with NO special tokens."""
        
        # Pre-tokenize the document
        pretokens_list = re.findall(self.PAT, document)
        
        # Convert pretokens to list of bytes
        bytes_list = [list(x.encode("utf-8")) for x in pretokens_list]
        tokens_list = []
        for j, tokens in enumerate(bytes_list):
            if verbose:
                print("Compressing: ", tokens, f"[{pretokens_list[j]}]")
            tokens = self._encode_tokens(tokens, verbose=verbose)
            tokens_list.append(tokens)
            if verbose:
                print("Compressed: ", tokens)
                print("-"*100)

        # Flatten the list of lists
        return [item for sublist in tokens_list for item in sublist]
    
    def encode(self, text: str, verbose: bool = False) -> list[int]:
        """
        Given a string, return a list of token IDs.
        
        Args:
            text (str):
                - may or may not contain special tokens;
                - any special tokens should always be from self.special_tokens
            verbose (bool): whether to print verbose output
        
        Returns:
            list[int]: a list of token IDs
        """
        
        # Split the text into documents across special tokens
        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        pattern = re.compile(f"({pattern})")  # 👈 IMPORTANT: parentheses
        
        # Get documents 
        docs = re.split(pattern, text)
        
        ids = []
        for doc in docs:
            if doc in self.special_tokens:
                ids.append(self.inverted_vocab[doc.encode("utf-8")])
            else:
                ids.extend(self._encode_document(doc, verbose=verbose))
        # import ipdb; ipdb.set_trace()
        return ids

        
        # # Pre-tokenize the text
        # pretokens_list = re.findall(self.PAT, text)
        
        # # Convert pretokens to list of bytes
        # bytes_list = [list(x.encode("utf-8")) for x in pretokens_list]
        # tokens_list = []
        # for j, tokens in enumerate(bytes_list):
        #     if verbose:
        #         print("Compressing: ", tokens, f"[{pretokens_list[j]}]")
        #     tokens = self._encode_tokens(tokens, verbose=verbose)
        #     tokens_list.append(tokens)
        #     if verbose:
        #         print("Compressed: ", tokens)
        #         print("-"*100)

        # # Flatten the list of lists
        # return [item for sublist in tokens_list for item in sublist]
    
    def encode_iterable(self, iterable):
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This is required for memory-eﬀicient tokenization 
        of large files that we cannot directly load into memory.
        """
        pass

    
    def decode(self, ids: list[int], verbose: bool = False) -> str:
        tokens = b"".join([self.vocab[x] for x in ids])
        return tokens.decode("utf-8", errors="replace")



if __name__ == "__main__":
    data_dir = "/scratch/shared/beegfs/piyush/datasets/text_data"
    train_dataset = "TinyStoriesV2-GPT4-train"
    eval_dataset = "TinyStoriesV2-GPT4-valid"
    # dataset = "owt_train"
    filepath = f"{data_dir}/{train_dataset}-bpe_optimized.npz"
    textpath = f"{data_dir}/{eval_dataset}.txt"
    
    # Initialize tokenizer
    tokenizer = Tokenizer.from_file(filepath=filepath, special_tokens=['<|endoftext|>'])

    n_chunks = 4096
    iterable = iter_text_chunks(
        textpath,
        special_tokens=["<|endoftext|>"],
        num_boundaries=n_chunks,
    )
    ids = []
    for chunk in tqdm(iterable, desc="Processing chunks", total=n_chunks):
        ids.extend(tokenizer.encode(chunk, verbose=False))
    # Save ids
    print("Number of tokens: ", len(ids))
    ids = np.array(ids, dtype=np.uint16)
    save_path = f"{data_dir}/{eval_dataset}-tokenized.npz"
    np.savez(save_path, ids=ids)
    # np.load(save_path)['ids']