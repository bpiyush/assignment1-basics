import torch
import torch.nn as nn
import numpy as np
import einops
import matplotlib.pyplot as plt
from collections.abc import Callable, Iterable
from typing import Optional
import math


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


if __name__ == "__main__":
    import einops
    B, T, V = 4, 16, 50257
    # logits = torch.randn((B, T, V))
    # logits = torch.zeros((B, T, V))
    logits = torch.rand((B, T, V))
    targets = torch.randint(0, V, (B, T))
    # loss_ = cross_entropy_(logits, targets)
    loss = cross_entropy(
        einops.rearrange(logits, "b t v -> (b t) v"),
        einops.rearrange(targets, "b t -> (b t)"),
    )
    print("Tester loss: ", loss.item())
    # print("Tester loss: ", loss_.item())
    print("Random loss: ", torch.log(torch.tensor([V])).item())
    
    pplx = perplexity(logits, targets)
    print("Perplexity: ", pplx)
    