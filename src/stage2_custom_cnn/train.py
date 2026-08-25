"""
Custom CNN 모델 학습 스크립트 (RTX 5050 최적화)
- AMP (FP16 Mixed Precision) 가속 및 GradScaler 적용
- Cosine Annealing Learning Rate Scheduler
- 에포크별 Loss/Acc 모니터링, Early Stopping 및 Best Checkpoint 저장
- 순수 NumPy 기반 Confusion Matrix 및 Classification Report 생성 (외부 의존성 없음)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.stage1_data.dataset import get_dataloaders
from src.stage2_custom_cnn.model import CustomFootwearCNN

MODELS_DIR = ROOT_DIR / "models"


def compute_confusion_matrix(y_true, y_pred, num_classes: int):
    """순수 NumPy 기반 Confusion Matrix 계산"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def get_classification_report(cm: np.ndarray, class_names: list) -> str:
    """순수 NumPy 기반 정밀도, 재현율, F1-Score 리포트 생성"""
    lines = [
        f"{'Class':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
        "-" * 68
    ]
    precisions, recalls, f1s, supports = [], [], [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        sup = int(cm[i, :].sum())
        
        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)
        lines.append(f"{name:<18} | {prec:>8.2f}% | {rec:>8.2f}% | {f1:>8.2f}% | {sup:>8d}")
        
    lines.append("-" * 68)
    total_sup = sum(supports)
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    lines.append(f"{'Macro Avg':<18} | {macro_prec:>8.2f}% | {macro_rec:>8.2f}% | {macro_f1:>8.2f}% | {total_sup:>8d}")
    return "\n".join(lines)


def plot_metrics(history, save_path: Path):
    """학습 곡선(Loss / Accuracy) 시각화 저장"""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss 곡선
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r--s', label='Val Loss')
    ax1.set_title('Custom CNN: Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 2. Accuracy 곡선
    ax2.plot(epochs, history["train_acc"], 'b-o', label='Train Accuracy')
    ax2.plot(epochs, history["val_acc"], 'r--s', label='Val Accuracy')
    ax2.set_title('Custom CNN: Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📈 학습 지표 그래프 저장 완료: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Confusion Matrix 히트맵 생성 및 저장"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Custom CNN Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"🎯 Confusion Matrix 저장 완료: {save_path}")


def evaluate(model, dataloader, criterion, device):
    """검증 / 테스트 루프"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc, np.array(all_targets), np.array(all_preds)


def train_custom_cnn(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 5e-4,
    early_stopping_patience: int = 15
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("🚀 [2단계] Custom CNN 학습 시작 (RTX 5050 최적화 - Warmup + Cosine)")
    print(f" - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" - Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 60)

    # 1. DataLoader 로드
    loaders = get_dataloaders(batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    class_names = loaders["class_names"]

    # 2. 모델, 손실함수, 옵티마이저, Warmup+Cosine 스케줄러, AMP 스케일러 정의
    model = CustomFootwearCNN(num_classes=len(class_names), dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # 3 에포크 Linear Warmup 후 Cosine Annealing 전환
    warmup_epochs = min(3, epochs // 5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = MODELS_DIR / "custom_cnn_best.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP Mixed Precision 순전파
            with autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            # 스케일러 역전파 & 옵티마이저 스텝
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        # 검증 루프
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        lr_curr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {lr_curr:.6f}")
        
        # 최고 성능 체크포인트 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names
            }, best_model_path)
            print(f"  ⭐ Best Model 저장 완료 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early Stopping triggered (Epoch {epoch})")
                break

    total_time = time.time() - start_time
    print(f"\n🎉 Custom CNN 학습 완료! 소요 시간: {total_time:.1f}초")
    
    # 3. 학습 곡선 저장
    plot_metrics(history, MODELS_DIR / "custom_cnn_metrics.png")
    
    # 4. 최종 Test Set 평가
    print("\n🧪 Best Model로 Test Set 최종 평가...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    
    print("=" * 60)
    print(f"📊 Custom CNN 최종 Test 결과:")
    print(f" - Test Loss:     {test_loss:.4f}")
    print(f" - Test Accuracy: {test_acc:.2f}%")
    print("=" * 60)
    print("\n[분류 성능 리포트 (Classification Report)]")
    print(get_classification_report(cm, class_names))
    
    plot_confusion_matrix(cm, class_names, MODELS_DIR / "custom_cnn_confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom CNN on Footwear Dataset")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    args = parser.parse_args()
    
    train_custom_cnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
