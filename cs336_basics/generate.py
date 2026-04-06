"""Script to generate text from a trained model."""
import os
import torch
import torch.nn as nn
import einops

import argparse
import  cs336_basics.model as models
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import load_checkpoint, print_update, generate


def generate_samples(prompt="Hello! I am a language model ", n_samples=5):
    print_update("Generating sample text")
    model.eval()
    token_ids = torch.tensor(tokenizer.encode(prompt))
    token_ids = einops.repeat(token_ids, "t -> b t", b=n_samples).to(device)
    token_positions = torch.arange(0, token_ids.shape[1], device=token_ids.device)
    token_positions = einops.repeat(token_positions, "t -> b t", b=token_ids.shape[0]).to(device)
    output = generate(
        model,
        token_ids,
        token_positions,
        max_new_tokens=64,
        endoftext_index=tokenizer.encode(tokenizer.special_tokens[0])[0],
        context_length=args.context_length,
        softmax_temp=1.,
        top_p=0.95,
        # top_p=None,
    )
    print("\n--------------------------------\n".join([tokenizer.decode(o.tolist()) for o in output]))
    return output


if __name__ == "__main__":
    DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets/text_data"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tok_path", type=str, default=f"{DATA_ROOT}/TinyStoriesV2-GPT4-train-bpe_optimized.npz",
    )
    # Model
    parser.add_argument("--vocab_size", type=int, default=10257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=1e4)
    parser.add_argument("--layer_norm", type=str, default="pre", choices=["pre", "post", "none"])
    parser.add_argument("--ffn_act", type=str, default="swiglu", choices=["swiglu", "silu"])
    parser.add_argument("--no_rope", action="store_true")
    parser.add_argument("--ckpt_name", type=str, default="train_gpt2_17M_20260405_182654")
    args = parser.parse_args()
    
    device = "cuda:0"
    ckpt_root = "/work/piyush/experiments/cs336/assignment1"
    
    # Define model
    print_update("Loading model", color='yellow')
    model = models.Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        theta=args.theta,
        layer_norm=args.layer_norm,
        ffn_act=args.ffn_act,
        use_rope=not args.no_rope,
    )
    model = model.to(device)
    models.count_params(model)
    
    # Load checkpoint
    ckpt_path = f"{ckpt_root}/{args.ckpt_name}/best_model.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint file {ckpt_path} does not exist"
    load_checkpoint(ckpt_path, model)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(filepath=args.tok_path, special_tokens=['<|endoftext|>'])
    
    # Generate samples
    prompt = "Once upon a time, there was a pretty girl named"
    prompt = "Hello! I am a language model"
    # prompt = "This is a story about Sam Altman."
    generate_samples(prompt=prompt, n_samples=5)    
    import ipdb; ipdb.set_trace()
