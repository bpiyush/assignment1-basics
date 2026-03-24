import numpy as np
from abc import ABC
from termcolor import colored
import random
import regex as re
from cs336_basics.pretokenization import (
    PAT as REGEX_PATTERN,
    # find_chunk_boundaries,
    # pretokenize_chunk,
    # pretokenize_document,
)


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
    
    def encode(self, text: str, verbose: bool = False) -> list[int]:
        
        # Pre-tokenize the text
        pretokens_list = re.findall(self.PAT, text)
        
        # Convert pretokens to list of bytes
        bytes_list = [list(x.encode("utf-8")) for x in pretokens_list]
        tokens_list = []
        for j, tokens in enumerate(bytes_list):
            if verbose:
                print("Compressing: ", tokens, f"[{pretokens_list[j]}]")
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
            tokens_list.append(tokens)
            if verbose:
                print("Compressed: ", tokens)
                print("-"*100)

        # Flatten the list of lists
        return [item for sublist in tokens_list for item in sublist]
    
    def decode(self, ids: list[int], verbose: bool = False) -> str:
        tokens = b"".join([self.vocab[x] for x in ids])
        return tokens.decode("utf-8", errors="replace")

