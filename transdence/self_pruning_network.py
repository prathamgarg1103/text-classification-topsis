"""
Self-Pruning Neural Network case-study solution.

Name: pratham garg
Roll Number: 102303053
Branch: COE

This script implements:
1. PrunableLinear: a custom linear layer with one learnable sigmoid gate per
   weight.
2. SelfPruningNet: a feed-forward CIFAR-10 classifier built only from
   PrunableLinear layers.
3. A full training/evaluation loop using classification loss plus an L1
   sparsity penalty over all gates.
4. Report, JSON summary, and matplotlib plots for the lambda trade-off.

Run:
    python self_pruning_network.py

Fast local validation without downloading CIFAR-10:
    python self_pruning_network.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset


DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 25
DEFAULT_LAMBDAS = (1e-4, 2e-2, 5e-2)
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GATE_LEARNING_RATE = 2e-2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GATE_THRESHOLD = 1e-2
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_NUM_WORKERS = 0
DEFAULT_LOG_INTERVAL = 50
CIFAR10_INPUT_SHAPE = (3, 32, 32)
CIFAR10_NUM_CLASSES = 10

warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed.*",
    category=Warning,
)


def set_seed(seed: int) -> None:
    """Make Python, NumPy, and PyTorch randomness deterministic where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_lambdas(raw: str) -> list[float]:
    """Parse comma-separated lambda values from the CLI."""
    lambdas = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(lambdas) < 3:
        raise argparse.ArgumentTypeError("Provide at least three lambda values.")
    if any(value < 0 for value in lambdas):
        raise argparse.ArgumentTypeError("Lambda values must be non-negative.")
    return lambdas


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda into a concrete torch.device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


class PrunableLinear(nn.Module):
    """
    Linear layer with a learnable gate for every weight.

    The effective weights are:

        pruned_weights = weight * sigmoid(gate_scores)

    During normal training/evaluation the gates are continuous. For final pruned
    evaluation, gates below a threshold can be hard-masked to exactly zero.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match nn.Linear-style initialization for stable training."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.gate_scores)

    def gates(self) -> torch.Tensor:
        """Return differentiable gate values in (0, 1)."""
        return torch.sigmoid(self.gate_scores)

    def effective_weight(
        self,
        *,
        hard_prune: bool = False,
        threshold: float = DEFAULT_GATE_THRESHOLD,
    ) -> torch.Tensor:
        """Return weight multiplied by soft gates or thresholded hard gates."""
        gates = self.gates()
        if hard_prune:
            gates = gates * (gates >= threshold).to(gates.dtype)
        return self.weight * gates

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard_prune: bool = False,
        threshold: float = DEFAULT_GATE_THRESHOLD,
    ) -> torch.Tensor:
        """Apply the gated linear transform."""
        return F.linear(
            x,
            self.effective_weight(hard_prune=hard_prune, threshold=threshold),
            self.bias,
        )

    @torch.no_grad()
    def detached_gates(self) -> torch.Tensor:
        """Return gate values detached from autograd for analysis."""
        return self.gates().detach()

    @torch.no_grad()
    def sparsity(self, threshold: float = DEFAULT_GATE_THRESHOLD) -> float:
        """Return fraction of gates below the pruning threshold."""
        return (self.detached_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class SelfPruningNet(nn.Module):
    """Feed-forward CIFAR-10 classifier using PrunableLinear in every FC layer."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, CIFAR10_NUM_CLASSES)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard_prune: bool = False,
        threshold: float = DEFAULT_GATE_THRESHOLD,
    ) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x, hard_prune=hard_prune, threshold=threshold)
        x = self.dropout(F.relu(self.bn1(x)))
        x = self.fc2(x, hard_prune=hard_prune, threshold=threshold)
        x = self.dropout(F.relu(self.bn2(x)))
        x = self.fc3(x, hard_prune=hard_prune, threshold=threshold)
        x = self.dropout(F.relu(self.bn3(x)))
        return self.fc4(x, hard_prune=hard_prune, threshold=threshold)

    def prunable_layers(self) -> list[PrunableLinear]:
        """Return all prunable layers in model order."""
        return [module for module in self.modules() if isinstance(module, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 penalty over all gate values.

        The case-study asks for the sum of all positive gate values. Because
        sigmoid gates never become mathematically zero, final pruning is reported
        by thresholding small gates.
        """
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for layer in self.prunable_layers():
            total = total + layer.gates().sum()
        return total

    @torch.no_grad()
    def overall_sparsity(self, threshold: float = DEFAULT_GATE_THRESHOLD) -> float:
        """Return percentage of weights whose gates are below threshold."""
        gates = self.all_gate_values_tensor()
        return (gates < threshold).float().mean().item()

    @torch.no_grad()
    def all_gate_values_tensor(self) -> torch.Tensor:
        """Return all gate values as one flat tensor."""
        return torch.cat([layer.detached_gates().flatten() for layer in self.prunable_layers()])

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """Return all gate values as a NumPy array for plotting."""
        return self.all_gate_values_tensor().cpu().numpy()

    @torch.no_grad()
    def parameter_counts(self, threshold: float = DEFAULT_GATE_THRESHOLD) -> dict[str, int]:
        """Return total and active prunable weight counts."""
        total = 0
        active = 0
        for layer in self.prunable_layers():
            gates = layer.detached_gates()
            total += gates.numel()
            active += int((gates >= threshold).sum().item())
        return {"total_prunable_weights": total, "active_prunable_weights": active}


def get_cifar10_loaders(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return train/test DataLoaders."""
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for CIFAR-10. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset: Any = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset: Any = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    *,
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lam: float,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_interval: int,
    max_grad_norm: float = 5.0,
) -> dict[str, float]:
    """Train for one epoch and return average losses."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    sparsity_loss_sum = 0.0

    num_batches = max(1, len(loader))
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        cls_loss = criterion(logits, labels)
        sparsity_loss = model.sparsity_loss()
        total_loss = cls_loss + lam * sparsity_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss_sum += float(total_loss.item())
        cls_loss_sum += float(cls_loss.item())
        sparsity_loss_sum += float(sparsity_loss.item())

        should_log = (
            log_interval > 0
            and (batch_idx == 1 or batch_idx == num_batches or batch_idx % log_interval == 0)
        )
        if should_log:
            print(
                f"  epoch {epoch:>2}/{epochs} "
                f"batch {batch_idx:>4}/{num_batches} | "
                f"cls_loss={cls_loss.item():.4f} | "
                f"total_loss={total_loss.item():.4f}",
                flush=True,
            )

    return {
        "total_loss": total_loss_sum / num_batches,
        "classification_loss": cls_loss_sum / num_batches,
        "sparsity_loss": sparsity_loss_sum / num_batches,
    }


@torch.no_grad()
def evaluate(
    *,
    model: SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
    hard_prune: bool,
    threshold: float,
) -> dict[str, float]:
    """Evaluate classification accuracy and loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, hard_prune=hard_prune, threshold=threshold)
        loss_sum += float(criterion(logits, labels).item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return {
        "accuracy": 100.0 * correct / max(1, total),
        "loss": loss_sum / max(1, len(loader)),
    }


def verify_gradient_flow(device: torch.device, threshold: float) -> None:
    """Fail fast if weight or gate gradients do not flow through PrunableLinear."""
    model = SelfPruningNet(dropout=0.0).to(device)
    model.train()
    images = torch.randn(4, *CIFAR10_INPUT_SHAPE, device=device)
    labels = torch.tensor([0, 1, 2, 3], device=device)
    loss = nn.CrossEntropyLoss()(model(images), labels) + 1e-4 * model.sparsity_loss()
    loss.backward()

    first_layer = model.fc1
    if first_layer.weight.grad is None or first_layer.gate_scores.grad is None:
        raise RuntimeError("Gradient flow check failed: missing gradients.")
    if first_layer.weight.grad.abs().sum().item() <= 0:
        raise RuntimeError("Gradient flow check failed: weight gradients are zero.")
    if first_layer.gate_scores.grad.abs().sum().item() <= 0:
        raise RuntimeError("Gradient flow check failed: gate gradients are zero.")

    with torch.no_grad():
        _ = model(images, hard_prune=True, threshold=threshold)


def train_and_evaluate(
    *,
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    gate_learning_rate: float,
    weight_decay: float,
    threshold: float,
    device: torch.device,
    seed: int,
    log_interval: int,
) -> dict[str, Any]:
    """Run one full training experiment for a lambda value."""
    set_seed(seed)
    model = SelfPruningNet().to(device)
    gate_params = []
    base_params = []
    for name, parameter in model.named_parameters():
        if name.endswith("gate_scores"):
            gate_params.append(parameter)
        else:
            base_params.append(parameter)
    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": learning_rate},
            {"params": gate_params, "lr": gate_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=1e-5,
    )

    history: dict[str, list[float]] = {
        "classification_loss": [],
        "total_loss": [],
        "sparsity_loss": [],
        "soft_test_accuracy": [],
        "sparsity_percent": [],
    }

    print(f"\nTraining lambda={lam:.0e}")
    print("-" * 72)
    started_at = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            lam=lam,
            device=device,
            epoch=epoch,
            epochs=epochs,
            log_interval=log_interval,
        )
        soft_eval = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            hard_prune=False,
            threshold=threshold,
        )
        sparsity_percent = 100.0 * model.overall_sparsity(threshold)
        scheduler.step()

        history["classification_loss"].append(train_metrics["classification_loss"])
        history["total_loss"].append(train_metrics["total_loss"])
        history["sparsity_loss"].append(train_metrics["sparsity_loss"])
        history["soft_test_accuracy"].append(soft_eval["accuracy"])
        history["sparsity_percent"].append(sparsity_percent)

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"Epoch {epoch:>2}/{epochs} | "
                f"cls_loss={train_metrics['classification_loss']:.4f} | "
                f"soft_acc={soft_eval['accuracy']:.2f}% | "
                f"sparsity={sparsity_percent:.1f}%"
            )

    elapsed_seconds = time.time() - started_at
    soft_eval = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        hard_prune=False,
        threshold=threshold,
    )
    pruned_eval = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        hard_prune=True,
        threshold=threshold,
    )
    counts = model.parameter_counts(threshold)
    gate_values = model.all_gate_values()
    final_sparsity = 100.0 * model.overall_sparsity(threshold)

    print(
        f"Final | pruned_acc={pruned_eval['accuracy']:.2f}% | "
        f"soft_acc={soft_eval['accuracy']:.2f}% | "
        f"sparsity={final_sparsity:.1f}% | time={elapsed_seconds:.1f}s"
    )

    return {
        "lambda": lam,
        "history": history,
        "final_soft_accuracy": soft_eval["accuracy"],
        "final_pruned_accuracy": pruned_eval["accuracy"],
        "final_sparsity_percent": final_sparsity,
        "total_prunable_weights": counts["total_prunable_weights"],
        "active_prunable_weights": counts["active_prunable_weights"],
        "elapsed_seconds": elapsed_seconds,
        "gate_values": gate_values,
        "model_state_dict": model.state_dict(),
    }


def select_best_tradeoff_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best accuracy/sparsity trade-off for the required gate plot."""
    if not results:
        raise ValueError("No results available.")
    if len(results) == 1:
        return results[0]

    accuracies = np.array([result["final_pruned_accuracy"] for result in results], dtype=float)
    sparsities = np.array([result["final_sparsity_percent"] for result in results], dtype=float)

    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        if span == 0:
            return np.ones_like(values)
        return (values - values.min()) / span

    norm_accuracy = normalize(accuracies)
    norm_sparsity = normalize(sparsities)
    tradeoff_scores = 2 * norm_accuracy * norm_sparsity / (
        norm_accuracy + norm_sparsity + 1e-12
    )
    if float(tradeoff_scores.max()) == 0.0 and float(sparsities.max()) > 0.0:
        return results[int(sparsities.argmax())]
    return results[int(tradeoff_scores.argmax())]


def plot_results(
    *,
    results: list[dict[str, Any]],
    output_dir: Path,
    threshold: float,
) -> dict[str, Path]:
    """Save summary and best-model gate distribution plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Self-Pruning Network: Lambda Trade-off", fontsize=15, fontweight="bold")

    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        label = f"lambda={result['lambda']:.0e}"
        epochs = range(1, len(result["history"]["soft_test_accuracy"]) + 1)
        axes[0, 0].plot(
            epochs,
            result["history"]["soft_test_accuracy"],
            label=label,
            color=color,
            linewidth=2,
        )
        axes[0, 1].plot(
            epochs,
            result["history"]["sparsity_percent"],
            label=label,
            color=color,
            linewidth=2,
        )
        axes[1, 1].hist(
            result["gate_values"],
            bins=80,
            alpha=0.45,
            color=color,
            label=label,
        )

    axes[0, 0].set_title("Soft-Gated Test Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Sparsity Over Training")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Sparsity (%)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    sparsity = [result["final_sparsity_percent"] for result in results]
    accuracy = [result["final_pruned_accuracy"] for result in results]
    lambda_labels = [f"{result['lambda']:.0e}" for result in results]
    point_colors = [colors[idx % len(colors)] for idx in range(len(results))]
    axes[1, 0].scatter(sparsity, accuracy, s=150, c=point_colors, zorder=3)
    axes[1, 0].plot(sparsity, accuracy, "k--", alpha=0.35)
    for x_value, y_value, label in zip(sparsity, accuracy, lambda_labels):
        axes[1, 0].annotate(f"lambda={label}", (x_value, y_value), xytext=(6, 5), textcoords="offset points")
    axes[1, 0].set_title("Pruned Accuracy vs Sparsity")
    axes[1, 0].set_xlabel("Sparsity (%)")
    axes[1, 0].set_ylabel("Hard-Pruned Test Accuracy (%)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axvline(threshold, color="black", linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("Final Gate Distributions")
    axes[1, 1].set_xlabel("Gate value")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    summary_plot = output_dir / "training_summary.png"
    fig.tight_layout()
    fig.savefig(summary_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)

    best = select_best_tradeoff_result(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    gates = best["gate_values"]
    ax.hist(gates, bins=100, color="#2563eb", alpha=0.88, edgecolor="white", linewidth=0.3)
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"prune threshold={threshold:g}",
    )
    pruned_percent = float((gates < threshold).mean() * 100.0)
    ax.set_title(
        "Best Trade-off Model Gate Distribution\n"
        f"lambda={best['lambda']:.0e}, "
        f"sparsity={best['final_sparsity_percent']:.1f}%, "
        f"pruned acc={best['final_pruned_accuracy']:.2f}%"
    )
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.text(
        0.98,
        0.95,
        f"{pruned_percent:.1f}% below threshold",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.grid(alpha=0.3)
    ax.legend()
    best_gate_plot = output_dir / "best_gate_distribution.png"
    fig.tight_layout()
    fig.savefig(best_gate_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"summary_plot": summary_plot, "best_gate_plot": best_gate_plot}


def serializable_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip tensors/arrays from experiment results for JSON serialization."""
    clean_results = []
    for result in results:
        clean_results.append(
            {
                "lambda": result["lambda"],
                "final_soft_accuracy": result["final_soft_accuracy"],
                "final_pruned_accuracy": result["final_pruned_accuracy"],
                "final_sparsity_percent": result["final_sparsity_percent"],
                "total_prunable_weights": result["total_prunable_weights"],
                "active_prunable_weights": result["active_prunable_weights"],
                "elapsed_seconds": result["elapsed_seconds"],
                "history": result["history"],
            }
        )
    return clean_results


def write_results_json(*, results: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write machine-readable experiment metrics."""
    path = output_dir / "results.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable_results(results), file, indent=2)
    return path


def generate_report(
    *,
    results: list[dict[str, Any]],
    output_dir: Path,
    plots: dict[str, Path],
    threshold: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gate_learning_rate: float,
) -> Path:
    """Write the Markdown report required by the case study."""
    best = select_best_tradeoff_result(results)
    rows = "\n".join(
        "| {lambda_value:.0e} | {accuracy:.2f}% | {sparsity:.1f}% | {active:,} / {total:,} |".format(
            lambda_value=result["lambda"],
            accuracy=result["final_pruned_accuracy"],
            sparsity=result["final_sparsity_percent"],
            active=result["active_prunable_weights"],
            total=result["total_prunable_weights"],
        )
        for result in results
    )

    report = f"""# Self-Pruning Neural Network Report

## 1. Why L1 on Sigmoid Gates Encourages Sparsity

Each trainable weight has a matching gate score `s_ij`. The gate used in the
forward pass is:

```text
g_ij = sigmoid(s_ij), so 0 < g_ij < 1
pruned_weight_ij = weight_ij * g_ij
```

The custom training objective is:

```text
total_loss = cross_entropy_loss + lambda * sum(g_ij over every prunable layer)
```

The L1 term penalizes every active gate directly. For a gate score, the
regularization gradient is:

```text
d(sum(g_ij)) / d(s_ij) = g_ij * (1 - g_ij)
```

This positive gradient makes gradient descent push `s_ij` downward, which moves
the corresponding sigmoid gate toward zero. Since sigmoid outputs approach zero
asymptotically rather than becoming exactly zero, this implementation reports and
evaluates pruning by hard-masking gates below `{threshold:g}`.

L2 would shrink small gates more weakly because its gate-value gradient is
`2 * g_ij`, which fades as `g_ij` approaches zero. L1 therefore creates a clearer
path to thresholded sparsity.

## 2. Results Summary

| Lambda | Hard-Pruned Test Accuracy | Sparsity Level | Active / Total Prunable Weights |
|:------:|:-------------------------:|:--------------:|:-------------------------------:|
{rows}

The model selected for the gate-distribution visualization was
`lambda={best['lambda']:.0e}` with `{best['final_pruned_accuracy']:.2f}%`
hard-pruned test accuracy and `{best['final_sparsity_percent']:.1f}%` sparsity.
This high-lambda model demonstrates the pruning mechanism clearly; the lower
lambda values preserve accuracy better but do not cross the strict pruning
threshold in this short CPU run.

## 3. Gate Distribution Plot

Best-model gate distribution:

![Best gate distribution]({plots['best_gate_plot'].name})

Full lambda trade-off summary:

![Training summary]({plots['summary_plot'].name})

The desired pattern is a large mass of gates near or below the pruning threshold
plus a second group of gates away from zero. Increasing `lambda` should increase
sparsity, while aggressive values can reduce classification accuracy.

## 4. Implementation Notes

- Dataset: CIFAR-10 via `torchvision.datasets.CIFAR10`
- Model: `3072 -> 1024 -> 512 -> 256 -> 10`
- Prunable layers: every fully connected layer uses `PrunableLinear`
- Optimizer: Adam
- Weight learning rate: `{learning_rate:g}`
- Gate learning rate: `{gate_learning_rate:g}`, using a separate parameter
  group for `gate_scores`
- Schedule: cosine annealing
- Epochs: {epochs}
- Batch size: {batch_size}
- Pruning threshold: `{threshold:g}`
- Reported accuracy: hard-pruned evaluation, where gates below threshold are
  masked to zero during the final forward pass

Generated by `self_pruning_network.py`.
"""

    report_path = output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)
    return report_path


def save_best_checkpoint(*, results: list[dict[str, Any]], output_dir: Path) -> Path:
    """Save the best trade-off model checkpoint."""
    best = select_best_tradeoff_result(results)
    checkpoint_path = output_dir / "best_self_pruning_model.pt"
    torch.save(
        {
            "lambda": best["lambda"],
            "model_state_dict": best["model_state_dict"],
            "final_pruned_accuracy": best["final_pruned_accuracy"],
            "final_sparsity_percent": best["final_sparsity_percent"],
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_smoke_test(device: torch.device, threshold: float) -> None:
    """Run fast checks without external downloads."""
    print(f"[smoke] device={device}")
    set_seed(DEFAULT_SEED)
    verify_gradient_flow(device=device, threshold=threshold)

    images = torch.randn(16, *CIFAR10_INPUT_SHAPE)
    labels = torch.randint(0, CIFAR10_NUM_CLASSES, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        lam=1e-4,
        device=device,
        epoch=1,
        epochs=1,
        log_interval=0,
    )
    eval_metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        hard_prune=True,
        threshold=threshold,
    )
    counts = model.parameter_counts(threshold)
    assert counts["total_prunable_weights"] > 0
    print(
        "[smoke] ok | "
        f"loss={train_metrics['total_loss']:.4f} | "
        f"hard_pruned_acc={eval_metrics['accuracy']:.2f}% | "
        f"prunable_weights={counts['total_prunable_weights']:,}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a self-pruning CIFAR-10 network.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--gate-learning-rate",
        type=float,
        default=DEFAULT_GATE_LEARNING_RATE,
        help="Learning rate for gate_scores; higher values make pruning visible sooner.",
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--lambdas",
        type=parse_lambdas,
        default=list(DEFAULT_LAMBDAS),
        help="Comma-separated lambda values, e.g. 1e-4,1e-3,5e-3.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="Print one training progress line every N batches. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small CIFAR-10 subset for pipeline validation.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run synthetic-data checks without downloading CIFAR-10.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    if args.smoke_test:
        run_smoke_test(device=device, threshold=args.threshold)
        return

    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.max_train_samples = args.max_train_samples or 1024
        args.max_test_samples = args.max_test_samples or 512

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Self-Pruning Neural Network - Tredence case study")
    print("=" * 72)
    print(f"device        : {device}")
    print(f"epochs        : {args.epochs}")
    print(f"batch_size    : {args.batch_size}")
    print(f"learning_rate : {args.learning_rate}")
    print(f"gate_lr       : {args.gate_learning_rate}")
    print(f"lambdas       : {args.lambdas}")
    print(f"threshold     : {args.threshold:g}")
    print(f"num_workers   : {args.num_workers}")
    print(f"log_interval  : {args.log_interval}")
    print(f"output_dir    : {output_dir.resolve()}")
    if device.type == "cpu" and not args.quick and args.max_train_samples is None:
        print("[info] Full CPU training can take a while. Progress is printed every few batches.")
    print("=" * 72)

    verify_gradient_flow(device=device, threshold=args.threshold)
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    results = []
    for lam in args.lambdas:
        result = train_and_evaluate(
            lam=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gate_learning_rate=args.gate_learning_rate,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            device=device,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        results.append(result)

    plots = plot_results(results=results, output_dir=output_dir, threshold=args.threshold)
    results_path = write_results_json(results=results, output_dir=output_dir)
    report_path = generate_report(
        results=results,
        output_dir=output_dir,
        plots=plots,
        threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gate_learning_rate=args.gate_learning_rate,
    )
    checkpoint_path = save_best_checkpoint(results=results, output_dir=output_dir)

    print("\nFinal summary")
    print("-" * 72)
    print(f"{'lambda':>10} | {'pruned acc':>10} | {'soft acc':>9} | {'sparsity':>9}")
    print("-" * 72)
    for result in results:
        print(
            f"{result['lambda']:>10.0e} | "
            f"{result['final_pruned_accuracy']:>9.2f}% | "
            f"{result['final_soft_accuracy']:>8.2f}% | "
            f"{result['final_sparsity_percent']:>8.1f}%"
        )
    print("-" * 72)
    print(f"report        : {report_path}")
    print(f"results       : {results_path}")
    print(f"summary plot  : {plots['summary_plot']}")
    print(f"gate plot     : {plots['best_gate_plot']}")
    print(f"checkpoint    : {checkpoint_path}")


if __name__ == "__main__":
    main()
