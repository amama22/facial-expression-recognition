
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import wandb

import data as data_mod
import models as models_mod
import utils as utils_mod

ARCHITECTURES = {
    "TinyCNN": models_mod.TinyCNN,
}


def build_model(config, device):
    arch = ARCHITECTURES[config["arch"]]
    return arch(num_classes=data_mod.NUM_CLASSES).to(device)


def build_optimizer(model, config):
    wd = config.get("weight_decay", 0.0)
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=wd)
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {config['optimizer']}")


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        loss_sum += loss.item() * xb.size(0)
        preds = out.argmax(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    macro_f1 = f1_score(targets, preds, average="macro")
    return loss_sum / total, correct / total, macro_f1, preds, targets


def run_experiment(config, csv_path, project="fer2013", run_name=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils_mod.set_seed(config.get("seed", 42))

    train_loader, val_loader, test_loader = data_mod.get_dataloaders(
        csv_path,
        batch_size=config["batch_size"],
        augment=config.get("augment", False),
        use_weighted_sampler=config.get("weighted_sampler", False),
    )

    model = build_model(config, device)
    optimizer = build_optimizer(model, config)

    if config.get("class_weighted_loss", False):
        _, labels, usage = data_mod.load_fer_arrays(csv_path)
        weight = data_mod.class_weights(labels[usage == "Training"]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    run = wandb.init(
        project=project,
        name=run_name or f"{config['arch']}_lr{config['lr']}_bs{config['batch_size']}",
        group=config["arch"],
        config=config,
        reinit=True,
    )

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)

        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": val_loss, "val/acc": val_acc, "val/macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"epoch {epoch:02d}  train_loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val_loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(
        model, test_loader, device, criterion)

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["test_acc"] = test_acc
    wandb.summary["test_macro_f1"] = test_f1

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=test_targets.tolist(),
            preds=test_preds.tolist(),
            class_names=[data_mod.EMOTIONS[i] for i in range(data_mod.NUM_CLASSES)],
        )
    })

    print(f"\nTEST  acc {test_acc:.3f}  macro_f1 {test_f1:.3f}")

    wandb.finish()

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_macro_f1": test_f1
    }
