import torch
import torch.nn as nn
import numpy as np
import einops
import matplotlib.pyplot as plt


# def cross_entropy_(logits, targets):
#     """
#     Computes cross-entropy loss for language modelling.
    
#     Args:
#         logits (torch.Tensor): [B, T, V]
#         targets (torch.Tensor): [B, T]
#     """
#     B, T, V = logits.shape
#     logits_max = logits.max(dim=-1, keepdim=False).values # [B, T]
#     logits_hat = logits[torch.arange(B)[:, None], torch.arange(T)[None, :], targets] # [B, T]
#     loss = ((logits_max - logits_hat) + torch.log(torch.exp(logits - logits_max[:, :, None]).sum(dim=-1))).mean()
#     return loss


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
    