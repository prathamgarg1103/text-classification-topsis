# Self-Pruning Neural Network

**Name:** pratham garg
**Roll Number:** 102303053
**Branch:** COE

Tredence AI Engineer case-study solution for a CIFAR-10 classifier that learns
to prune its own weights during training.

## Deliverables

- `self_pruning_network.py`: complete implementation, training loop, evaluation,
  plotting, and report generation.
- `report.md`: generated after a full run with the lambda comparison table.
- `best_gate_distribution.png`: generated plot for the best accuracy/sparsity
  trade-off model.
- `training_summary.png`: generated plot comparing all lambda runs.
- `results.json`: generated machine-readable experiment metrics.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Fast Validation

This checks model shape, gradient flow through `weight` and `gate_scores`, hard
pruned evaluation, and one synthetic training step without downloading CIFAR-10.

```powershell
python self_pruning_network.py --smoke-test
```

If you are on Windows or CPU-only PyTorch, the script defaults to
`--num-workers 0` and prints batch progress so it does not look frozen.

## Full Case-Study Run

```powershell
python self_pruning_network.py
```

The default run trains three lambda values: `1e-4`, `2e-2`, and `5e-2`.
The script uses a separate gate learning rate so pruning becomes visible without
needing an extremely long CPU run.

The included `report.md` and plots were generated on CPU with:

```powershell
python self_pruning_network.py --epochs 3 --output-dir .
```

Useful quick pipeline check with CIFAR-10:

```powershell
python self_pruning_network.py --quick --output-dir outputs
```

## Notes

- Final reported accuracy is hard-pruned accuracy: gates below `1e-2` are masked
  to zero during evaluation.
- The L1 sparsity loss is the sum of sigmoid gate values across all
  `PrunableLinear` layers, matching the case-study specification.
- Use `--output-dir outputs` if you want generated artifacts kept out of the
  repository root.
