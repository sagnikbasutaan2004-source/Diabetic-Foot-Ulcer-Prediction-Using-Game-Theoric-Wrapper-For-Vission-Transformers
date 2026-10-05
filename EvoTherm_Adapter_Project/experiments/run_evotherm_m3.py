import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(r"d:\Diabetic Foot Ulcer Final\ThermoDataBase")
RESULTS_DIR = ROOT / "results_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _aug(split: str, img_size: int = 224):
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(noise_scale_factor=0.05, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class _ThermoDS(Dataset):
    def __init__(self, samples, split, img_size=224):
        self.samples = samples
        self.transform = _aug(split, img_size)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        aug = self.transform(image=image)
        return aug['image'], torch.tensor(label, dtype=torch.long)


def get_m3_splits(img_size: int = 224, batch_size: int = 8, num_workers: int = 2):
    """Collect samples from audited data root and return (train_dl, val_dl, test_dl)."""
    samples = []
    for label, group_name in [(1, 'DM Group'), (0, 'Control Group')]:
        group_dir = DATA_ROOT / group_name
        if not group_dir.exists():
            raise FileNotFoundError(f"Data root not found: {group_dir}")
        for subj_dir in sorted(group_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            subject_id = subj_dir.name
            for side in ['L', 'R']:
                img_path = subj_dir / f"{subject_id}_{side}.png"
                if img_path.exists():
                    samples.append((img_path, label, subject_id))

    assert len(samples) > 0, f"No thermal images found in {DATA_ROOT}"
    print(f"  [Data] Found {len(samples)} thermal images from {DATA_ROOT.parent.name}")

    subjects = [s[2] for s in samples]
    labels = [s[1] for s in samples]
    indices = list(range(len(samples)))

    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(indices, labels, subjects))

    val_idx = splits[0][1]
    test_idx = splits[1][1]
    train_idx = np.concatenate([splits[i][1] for i in range(2, 5)])

    # Verify zero subject leakage
    train_subjs = set(samples[i][2] for i in train_idx)
    val_subjs = set(samples[i][2] for i in val_idx)
    test_subjs = set(samples[i][2] for i in test_idx)
    assert len(train_subjs & val_subjs) == 0, "LEAKAGE: Train/Val subject overlap!"
    assert len(train_subjs & test_subjs) == 0, "LEAKAGE: Train/Test subject overlap!"
    assert len(val_subjs & test_subjs) == 0, "LEAKAGE: Val/Test subject overlap!"
    print(f"  [Leakage] PASSED: {len(train_subjs)} train / {len(val_subjs)} val / {len(test_subjs)} test subjects")
    print(f"  [Split] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} samples")

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_dl = DataLoader(_ThermoDS(train_samples, 'train', img_size),
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(_ThermoDS(val_samples, 'val', img_size),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(_ThermoDS(test_samples, 'test', img_size),
                         batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (15-bin) on raw softmax probabilities."""
    confs, preds = probs.max(dim=1)
    correct = (preds == labels).float()
    ece = 0.0
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= low) & (confs < high)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = confs[mask].mean().item()
        ece += mask.float().mean().item() * abs(bin_acc - bin_conf)
    return ece


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def _epoch_train(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss, n = 0.0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def _epoch_eval(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu()
        all_probs.append(probs)
        all_labels.append(y)
    probs_t = torch.cat(all_probs)
    labels_t = torch.cat(all_labels)
    return probs_t, labels_t


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute full classification metrics from probability tensor."""
    preds = probs.argmax(dim=1)
    preds_np = preds.numpy()
    labels_np = labels.numpy()
    probs_np = probs.numpy()

    acc = accuracy_score(labels_np, preds_np) * 100
    f1 = f1_score(labels_np, preds_np, average='macro') * 100
    try:
        auc = roc_auc_score(labels_np, probs_np[:, 1]) if probs_np.shape[1] == 2 else 0.0
    except Exception:
        auc = 0.0
    ece = compute_ece(probs, labels)
    cm = confusion_matrix(labels_np, preds_np)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        'accuracy': round(acc, 4),
        'f1_macro': round(f1, 4),
        'auc_roc': round(auc, 4),
        'ece': round(ece, 4),
        'sensitivity': round(sensitivity * 100, 4),
        'specificity': round(specificity * 100, 4),
        'confusion_matrix': cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Two-Phase Training (for adapted models)
# ---------------------------------------------------------------------------

def train_two_phase(model, train_dl, val_dl, device,
                    epochs: int = 20, lr: float = 1e-4,
                    warm_epochs: int = 5, patience: int = 8,
                    is_adapted: bool = True,
                    save_path: Path = None,
                    model_name: str = ''):
    """
    Two-phase training:
      Phase 1 (0..warm_epochs): backbone frozen, adapter lr.
      Phase 2 (warm_epochs..): backbone unfrozen at lr*0.1.
    """
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_acc = -1.0
    best_state = None
    no_improve = 0

    history = {'train_loss': [], 'val_acc': [], 'val_ece': []}

    for epoch in range(epochs):
        # Phase switching
        if is_adapted and hasattr(model, 'freeze_backbone'):
            if epoch == 0:
                model.freeze_backbone()
                optim_params = [
                    {'params': model.adapter_parameters(), 'lr': lr}
                ]
                optimizer = AdamW(optim_params, weight_decay=0.05)
                scheduler = CosineAnnealingLR(optimizer, T_max=warm_epochs, eta_min=lr*0.1)
                phase = 1
                print(f"    [Phase 1] Backbone FROZEN — training adapter + head only")

            elif epoch == warm_epochs:
                model.unfreeze_backbone()
                optimizer = AdamW([
                    {'params': model.adapter_parameters(), 'lr': lr},
                    {'params': model.backbone_parameters(), 'lr': lr * 0.1},
                ], weight_decay=0.05)
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warm_epochs, eta_min=1e-6)
                phase = 2
                no_improve = 0  # reset patience for phase 2
                print(f"    [Phase 2] Backbone UNFROZEN — joint fine-tuning at LR {lr*0.1:.2e}")
        elif epoch == 0:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        train_loss = _epoch_train(model, train_dl, optimizer, device, scaler)
        scheduler.step()

        val_probs, val_labels = _epoch_eval(model, val_dl, device)
        val_acc = accuracy_score(val_labels.numpy(), val_probs.argmax(1).numpy()) * 100
        val_ece = compute_ece(val_probs, val_labels)

        history['train_loss'].append(round(train_loss, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['val_ece'].append(round(val_ece, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if save_path is not None:
                torch.save({'model_state': best_state, 'val_acc': best_val_acc,
                            'epoch': epoch}, save_path)
        else:
            no_improve += 1

        print(f"    Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val ECE: {val_ece:.4f} | Best: {best_val_acc:.2f}%")

        if no_improve >= patience:
            print(f"    [EarlyStopping] No improvement for {patience} epochs. Stopping.")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# Build Models
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int = 2):
    """Build a single model by name."""
    from src.models.baselines import build_baseline
    from src.models.adapted_baselines import build_adapted_baseline

    if name.startswith('evo_'):
        return build_adapted_baseline(name, num_classes=num_classes)
    else:
        name_map = {'resnet50': 'resnet50', 'deit_tiny': 'deit_tiny', 'swinv2_tiny': 'swinv2_tiny'}
        return build_baseline(name_map[name], task='classification', num_classes=num_classes)


# ---------------------------------------------------------------------------
# Main Experiment Loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*65}")
    print(f"  EvoTherm-Adapter vs Baselines — M3 Plantar Thermal Classification")
    print(f"  Device: {device.upper()} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"{'='*65}\n")

    # Data
    img_size = 224
    train_dl, val_dl, test_dl = get_m3_splits(
        img_size=img_size, batch_size=args.batch_size
    )

    # Model list
    all_models = ['resnet50', 'deit_tiny', 'swinv2_tiny',
                  'evo_resnet50', 'evo_deit_tiny', 'evo_swinv2_tiny']
    if args.skip_baselines:
        all_models = [m for m in all_models if m.startswith('evo_')]
    if args.model:
        all_models = [args.model]

    ckpt_dir = RESULTS_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in all_models:
        print(f"\n{'─'*60}")
        print(f"  Training: {model_name.upper()}")
        print(f"{'─'*60}")

        model = build_model(model_name, num_classes=2).to(device)

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        adapter_params = 0
        if hasattr(model, 'adapter'):
            adapter_params = sum(p.numel() for p in model.adapter.parameters()) / 1e6
        print(f"  Total params: {total_params:.2f}M | Adapter params: {adapter_params:.2f}M")

        is_adapted = model_name.startswith('evo_')
        save_path = ckpt_dir / f"evotherm_m3_{model_name}_best.pt"

        history = train_two_phase(
            model, train_dl, val_dl, device,
            epochs=args.epochs, lr=args.lr,
            warm_epochs=args.warm_epochs,
            patience=args.patience,
            is_adapted=is_adapted,
            save_path=save_path,
            model_name=model_name,
        )

        # Test evaluation
        test_probs, test_labels = _epoch_eval(model, test_dl, device)
        metrics = compute_metrics(test_probs, test_labels)
        metrics['total_params_M'] = round(total_params, 3)
        metrics['adapter_params_M'] = round(adapter_params, 3)
        metrics['is_adapted'] = is_adapted
        metrics['history'] = history

        all_results[model_name] = metrics

        print(f"\n  [Test Results for {model_name}]")
        print(f"    Accuracy   : {metrics['accuracy']:.2f}%")
        print(f"    AUC-ROC    : {metrics['auc_roc']:.4f}")
        print(f"    ECE (raw)  : {metrics['ece']:.4f}")
        print(f"    Sensitivity: {metrics['sensitivity']:.2f}%")
        print(f"    Specificity: {metrics['specificity']:.2f}%")

    # Save JSON
    results_path = RESULTS_DIR / 'evotherm_m3_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  [OK] Full results saved → {results_path}")

    # Print comparison table
    print_summary_table(all_results)

    return all_results


def print_summary_table(results: dict):
    """Print a clean before/after comparison table."""
    print(f"\n{'='*85}")
    print(f"  EvoTherm-Adapter vs Baseline — Comparison Table (M3 Plantar Thermal)")
    print(f"{'='*85}")
    header = f"  {'Model':<20} {'Accuracy':>10} {'AUC-ROC':>9} {'ECE':>8} {'Sensitivity':>13} {'Params':>8}"
    print(header)
    print(f"  {'─'*80}")

    pairs = [
        ('resnet50',    'evo_resnet50'),
        ('deit_tiny',   'evo_deit_tiny'),
        ('swinv2_tiny', 'evo_swinv2_tiny'),
    ]
    for base_name, evo_name in pairs:
        for name, tag in [(base_name, 'Baseline'), (evo_name, '+EvoAdapter')]:
            if name not in results:
                continue
            m = results[name]
            label = f"{name} [{tag}]"
            print(f"  {label:<20} {m['accuracy']:>9.2f}% {m['auc_roc']:>9.4f} "
                  f"{m['ece']:>8.4f} {m['sensitivity']:>12.2f}% {m['total_params_M']:>7.2f}M")
        print(f"  {'─'*80}")

    # Save summary text
    summary_path = RESULTS_DIR / 'evotherm_m3_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EvoTherm-Adapter vs Baseline — M3 Plantar Thermal Classification\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':>10} {'AUC-ROC':>9} {'ECE':>8} {'Sensitivity':>13}\n")
        f.write("─"*80 + "\n")
        for name, m in results.items():
            tag = '+EvoAdapter' if m.get('is_adapted') else 'Baseline'
            f.write(f"{name+' ['+tag+']':<20} {m['accuracy']:>9.2f}% "
                    f"{m['auc_roc']:>9.4f} {m['ece']:>8.4f} {m['sensitivity']:>12.2f}%\n")
    print(f"\n  [OK] Summary table saved → {summary_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoTherm-Adapter M3 Experiment')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Total training epochs (default: 20)')
    parser.add_argument('--warm-epochs', type=int, default=5,
                        help='Phase 1 warm-up epochs with frozen backbone (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate for adapter (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience (default: 8)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Only train adapted models (skip standard baselines)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['resnet50', 'deit_tiny', 'swinv2_tiny',
                                 'evo_resnet50', 'evo_deit_tiny', 'evo_swinv2_tiny'],
                        help='Train a single model only')
    args = parser.parse_args()

    run_experiment(args)
