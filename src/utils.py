import math
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def initial_loss_check(model, loader, device, num_classes=7):
    model.eval()
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    loss = nn.functional.cross_entropy(model(xb), yb).item()
    expected = math.log(num_classes)
    print(f"initial loss = {loss:.3f}  (expected ~{expected:.3f})")
    return loss


def overfit_one_batch(model, loader, device, steps=200, lr=1e-3):
    model.train()
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        out = model(xb)
        loss = nn.functional.cross_entropy(out, yb)
        loss.backward()
        opt.step()
        if (i + 1) % 50 == 0:
            acc = (out.argmax(1) == yb).float().mean().item()
            print(f"step {i + 1:4d}  loss {loss.item():.4f}  acc {acc:.3f}")
    return loss.item()
