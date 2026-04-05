import os
import torch
import torch.nn as nn
import numpy as np
import einops
import matplotlib.pyplot as plt
from collections.abc import Callable, Iterable
from typing import Optional
import math
import cs336_basics.model as models
from cs336_basics.tokenizer import Tokenizer
import tqdm
import time


def cross_entropy(logits, targets):
    """
    Computes cross-entropy loss for language modelling.
    
    Args:
        logits (torch.Tensor): [B, V]
        targets (torch.Tensor): [B]
    """
    B, V = logits.shape
    logits_max = logits.max(dim=-1, keepdim=False).values # [B]
    logits_hat = logits[torch.arange(B), targets] # [B]
    loss = ((logits_max - logits_hat) + torch.log(torch.exp(logits - logits_max[:, None]).sum(dim=-1))).mean()
    return loss


def perplexity(logits, targets):
    """
    Computes cross-entropy loss for language modelling.
    
    Args:
        logits (torch.Tensor): [B, T, V]
        targets (torch.Tensor): [B, T]
    """
    B, T, V = logits.shape
    logits_max = logits.max(dim=-1, keepdim=False).values # [B, T]
    logits_hat = logits[torch.arange(B)[:, None], torch.arange(T)[None, :], targets] # [B, T]
    x = ((logits_max - logits_hat) + torch.log(torch.exp(logits - logits_max[:, :, None]).sum(dim=-1))) # [B, T]
    return torch.exp(x.mean(dim=-1)).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                else:
                    state = self.state[p] # Get state associated with p.
                    t = state.get("t", 1) # Get iteration number from the state, or initial value.
                    g = p.grad.data
                    m = state.get("m", torch.zeros_like(g))
                    v = state.get("v", torch.zeros_like(g))
                    
                    m = b1 * m + (1 - b1) * g
                    v = b2 * v + (1 - b2) * (g**2)
                    lr_t = lr * np.sqrt(1 - b2**t) / (1 - b1**t)
                    p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                    p.data *= (1 - lr * weight_decay)
                    
                    state["t"] = t + 1 # Increment iteration number.
                    state["m"] = m
                    state["v"] = v

        return loss


def cosine_learning_rate_schedule(t, lr_min, lr_max, iters_warmup, iters_cosine):
    if t < iters_warmup:
        lr = (t / iters_warmup) * lr_max
    elif t >= iters_warmup and t <= iters_cosine:
        angle = np.pi * (t - iters_warmup) / (iters_cosine - iters_warmup)
        lr = lr_min + 0.5 * (1 + np.cos(angle)) * (lr_max - lr_min)
    elif t > iters_cosine:
        lr = lr_min
    else:
        raise ValueError
    return lr


def compute_total_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-8) -> None:
    total_norm = compute_total_norm(parameters)
    if total_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data = p.grad.data * min(1.0, max_norm / (total_norm + eps))
    return total_norm


def get_batch(data, batch_size, context_length, device, deterministic=False):
    if deterministic:
        # Return the same batch every time
        ix = torch.arange(0, context_length * batch_size, context_length)
    else:
        ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def save_checkpoint(model, optimizer, iteration, out):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(data, out)


def load_checkpoint(src, model, optimizer=None):
    data = torch.load(src, weights_only=False)
    
    state_dict = {k.replace("_orig_mod.", ""):v for k, v in data["model"].items()}
    msg = model.load_state_dict(state_dict)
    print("Loaded checkpoint from ", src)
    print(msg)
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
    return data["iteration"]



def get_terminal_width():
    import shutil
    return shutil.get_terminal_size().columns


def print_update(update, fillchar=".", color="yellow", pos="left", **kwargs):
    from termcolor import colored
    # add ::: to the beginning and end of the update s.t. the total length of the
    # update spans the whole terminal
    try:
        terminal_width = get_terminal_width()
    except:
        terminal_width = 98
    if pos == "center":
        update = update.center(len(update) + 2, " ")
        update = update.center(terminal_width, fillchar)
    elif pos == "left":
        update = update.ljust(terminal_width, fillchar)
        update = update.ljust(len(update) + 2, " ")
    elif pos == "right":
        update = update.rjust(terminal_width, fillchar)
        update = update.rjust(len(update) + 2, " ")
    else:
        raise ValueError("pos must be one of 'center', 'left', 'right'")
    print(colored(update, color, **kwargs))


@torch.no_grad()
def generate(model, token_ids, token_positions, max_new_tokens, endoftext_index, context_length, softmax_temp=1., top_p=None):
    model.eval()

    x = token_ids
    p = token_positions
    
    for i in range(max_new_tokens):
        B, T = x.shape

        # Check context length
        if T <= context_length:
            x_cond = x
            p_cond = p
        else:
            x_cond = x[:, -context_length:]
            p_cond = p[:, -context_length:]
        
        # Compute logits
        o = model(x_cond, p_cond)
        Q = models.softmax(o[:, -1, :] / softmax_temp, dim=-1)
        
        if top_p is None:
            # Sample from the full distribution: [B,]
            x_next = torch.multinomial(Q, num_samples=1)
        else:
            # Run the top-p sampling procedure
            Q_sorted = Q.sort(dim=-1, descending=True)
            indices = torch.argmax((Q_sorted.values.cumsum(dim=-1) >= top_p).int(), dim=-1)
            x_next = []
            for b in range(B):
                token_indices_subset = Q_sorted.indices[b, :indices[b]+1]
                Q_subset = Q[b, token_indices_subset]
                Q_subset = Q_subset / Q_subset.sum()
                sampled_pos = torch.multinomial(Q_subset, num_samples=1)
                x_next.append(token_indices_subset[sampled_pos])
            x_next = torch.stack(x_next).to(x.device)
        p_next = torch.tensor([[T] for _ in range(B)], device=p.device, dtype=p.dtype)
        
        # Update x and p by appending generated token
        x = torch.cat([x, x_next], dim=-1)
        p = torch.cat([p, p_next], dim=-1)

        # TODO: If next generated token is <endoftext>, then break
        # Not sure how to do this at batch level (if only some part)
        # of the batch ends up with <endoftext>.

    return x


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(eval(f"data_{split}"), args.batch_size, args.context_length, device, deterministic=False)
            
            # Forward pass
            with torch.autocast(device_type=device, dtype=forward_dtype):
                logits = model(x, p)
                loss = cross_entropy(
                    einops.rearrange(logits, "b t d -> (b t) d"),
                    einops.rearrange(y, "b t -> (b t)"),
                )

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets/text_data"
    import argparse
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument(
        "--train_data", type=str, default=f"{DATA_ROOT}/owt_train-tokenized.npy",
    )
    parser.add_argument(
        "--valid_data", type=str, default=f"{DATA_ROOT}/owt_valid-tokenized.npy",
    )
    parser.add_argument(
        "--tok_path", type=str, default=f"{DATA_ROOT}/owt_train-bpe_optimized.npz",
    )
    # Model
    parser.add_argument("--vocab_size", type=int, default=32257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=1e4)
    parser.add_argument("--layer_norm", type=str, default="pre", choices=["pre", "post", "none"])
    parser.add_argument("--ffn_act", type=str, default="swiglu", choices=["swiglu", "silu"])
    parser.add_argument("--no_rope", action="store_true")
    # Optim
    # Batch size is tuned to a single 46-48GB GPU (e.g., A6000 or A40).
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    
    
    # Set env stuff
    device = "cuda:0"
    tf32_precision = "high"
    torch_compile = True
    forward_dtype = torch.bfloat16 # default: torch.float32

    lr_base = args.lr
    lr_min = lr_base / 10. # From Chinchilla paper.
    lr_max = lr_base
    
    # Tokens to process in a single step: closest nice number to 0.25M
    # (0.25M = 250,000); closest power of 2: 262,144
    tokens_per_step = 262144
    assert tokens_per_step % (args.context_length * args.batch_size) == 0, \
        f"context_length * batch_size must be a divisor of tokens_per_step "\
        f"(tokens_per_step = {tokens_per_step}\n"\
        f"context_length = {args.context_length}\n"\
        f"batch_size = {args.batch_size})"
    # This is number of sequences you can process on a single GPU at once.
    batch_size_micro = args.batch_size
    grad_accum_steps = tokens_per_step // (batch_size_micro * args.context_length)
    print(f"Batch size micro: {batch_size_micro}")
    print(f"Grad accum steps: {grad_accum_steps}")

    # Total tokens in OWT train set: 2,727,181,568
    # Closest power of 2: 2,684,354,560
    total_tokens_to_process = 2684354560
    max_iters = total_tokens_to_process // (args.batch_size * args.context_length * grad_accum_steps)
    if args.debug:
        max_iters = 50
    iters_warmup = int(max_iters * 0.1)
    iters_cosine = int(max_iters * 0.8)

    eval_iters = 25 # Number of batches to evaluate on and average over.
    eval_interval = 50 # Evaluate and log to W&B after this many iterations.

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    torch.set_float32_matmul_precision(tf32_precision)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"train_gpt2_17M_owt_{timestamp}"
    out_dir = f"/work/piyush/experiments/cs336/assignment1/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    if args.log_to_wandb:
        import wandb
        wandb.init(project="cs336_basics", name=run_name, entity="bpiyush")
        
        # Save config to W&B (dict.update returns None; build a merged dict)
        hparams = vars(args).copy()
        hparams.update(
            {
                "total_tokens_to_process": total_tokens_to_process,
                "max_iters": max_iters,
                "iters_warmup": iters_warmup,
                "iters_cosine": iters_cosine,
                "eval_iters": eval_iters,
                "eval_interval": eval_interval,
                "dataset": "openwebtext",
            }
        )
        wandb.config.update(hparams)
        
        # Also, save a mapping of the run_name to the hyperparameters locally
        save_dir = "./wandb_runs_hparams"
        import json
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{run_name}.json"), "w") as f:
            json.dump(hparams, f)
    
    # Load data
    print_update("Loading data", color='yellow')
    data_train = np.load(args.train_data, mmap_mode='r')
    data_valid = np.load(args.valid_data, mmap_mode='r')
    print(f"Number of train tokens: {len(data_train)/1e9:.2f} B")
    print(f"Number of valid tokens: {len(data_valid)/1e9:.2f} B")
    
    
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
    
    if torch_compile:
        print("Compiling model with torch.compile")
        t0 = time.time()
        model = torch.compile(model) # requires PyTorch 2.0
        t1 = time.time()
        print(f"Model compiled in {t1 - t0:.2f} seconds")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(filepath=args.tok_path, special_tokens=['<|endoftext|>'])
    
    # Inference: Test on a single batch
    test_single_batch = False
    if test_single_batch:
        print_update("Testing inference on a single batch")
        x, y = get_batch(
            data_train, args.batch_size, args.context_length, device=device
        )
        token_positions = torch.arange(0, x.shape[1], device=x.device, dtype=x.dtype)
        token_positions = einops.repeat(token_positions, "t -> b t", b=x.shape[0])
        with torch.no_grad():
            logits = model(x, token_positions)
        loss = cross_entropy(
            einops.rearrange(logits, "b t d -> (b t) d"),
            einops.rearrange(y, "b t -> (b t)"),
        )
        print("Loss on a single batch: ", loss.cpu().detach().item())
    
    # Generate sample text
    generate_sample_text = False
    if generate_sample_text:
        print_update("Generating sample text")
        prompt = "Hello! I am a language model "
        token_ids = torch.tensor(tokenizer.encode(prompt))
        token_ids = einops.repeat(token_ids, "t -> b t", b=5)
        token_positions = torch.arange(0, token_ids.shape[1], device=token_ids.device)
        token_positions = einops.repeat(token_positions, "t -> b t", b=token_ids.shape[0])
        output = generate(
            model,
            token_ids,
            token_positions,
            max_new_tokens=64,
            endoftext_index=tokenizer.encode(tokenizer.special_tokens[0])[0],
            context_length=args.context_length,
            softmax_temp=1.,
            top_p=0.7,
        )
        print("\n\n".join([tokenizer.decode(o.tolist()) for o in output]))
        import ipdb; ipdb.set_trace()
    
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=lr_base, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    
    # TODO
    # 1. Overfit on a single (deterministic) batch: DONE.
    # 2. Optimizations
        # 2.1. Use TF32 matrix multiplication: DONE.
        # 2.2. Use mixed-precision training (forward pass in bfloat16, backward pass in float32). DONE.
        # 2.3. torch.compile: DONE.
        # 2.4. Flash attention - not using it.
        # 2.5. Use nice numbers (e.g., 50304 for vocab size) - not using it.
        # 2.6. hyperparameters (gradient clipping, lr scheduling). DONE.
        # 2.7. Weight decay should only apply to matrices (not biases) - not using it.
        # 2.8. TODO Gradient accumulation.
        # 2.9. TODO Use multiple GPUs with DDP.
    # 3. TODO Add validation loss and perplexity.
    # 4. TODO Add logging to W&B
    # 5. TODO Add checkpointing
    
    # Training loop
    print_update("Training loop")
    pbar = tqdm.tqdm(range(max_iters), desc="Training", bar_format="{l_bar}{bar:20}{r_bar}")
    
    # Can fix the token positions during training (to save time)
    p = torch.arange(0, args.context_length, device=device)
    p = einops.repeat(p, "t -> b t", b=args.batch_size)
    
    best_val_loss = float('inf')

    for i in pbar:
        t0 = time.time()

        # Learning rate schedule
        lr = cosine_learning_rate_schedule(i, lr_min, lr_max, iters_warmup, iters_cosine)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # evaluate the loss on train/val sets
        if i % eval_interval == 0:
            # Time to evaluate the loss on train/val sets and log to W&B
            losses = estimate_loss()
            # print(f"step {i}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}")
            if args.log_to_wandb:
                wandb.log({
                    "train/loss": losses['train'],
                    "valid/loss": losses['valid'],
                    "lr": lr,
                    "iter": i,
                })
            
            # Save best model if validation loss is lower
            if losses['valid'] < best_val_loss and not args.no_save:
                print(f"Saving checkpoint with validation loss ({losses['valid']:.4f}) at step {i}:")
                best_val_loss = losses['valid']
                save_checkpoint(model, optimizer, i, f"{out_dir}/best_model.pth")

        # Accumulate gradients for grad_accum_steps
        loss_accum = 0.0
        for j in range(grad_accum_steps):

            # Get batch
            x, y = get_batch(data_train, args.batch_size, args.context_length, device, deterministic=False)

            # Set gradient to 0
            optimizer.zero_grad()
            
            # Forward pass
            with torch.autocast(device_type=device, dtype=forward_dtype):
                logits = model(x, p)
                loss = cross_entropy(
                    einops.rearrange(logits, "b t d -> (b t) d"),
                    einops.rearrange(y, "b t -> (b t)"),
                )
            
            # Important: Scale the loss by 1/grad_accum_steps
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            
            # Backward pass (accumulate gradients)
            loss.backward()
        
        # Gradient clipping after backward pass
        norm = gradient_clipping(model.parameters(), max_norm=1.0)
        
        # Update optimizer
        optimizer.step()
        
        # Wait for the GPU to finish its work before computing the time
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1e3 # in milliseconds
        
        tokens_per_sec = (args.batch_size * args.context_length * grad_accum_steps) / (t1 - t0)
        
        pbar.set_postfix(
            {
                "step": i,
                "loss": np.round(loss_accum, 5),
                "dt": f"{dt:.2f} ms",
                "t/s": f"{tokens_per_sec:.1f}",
                "norm": np.round(norm, 5),
                "lr": np.round(lr, 6),
            }
        )
        
        # Log to W&B
        


