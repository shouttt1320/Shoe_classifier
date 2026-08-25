"""
ResNet 전이학습 (Feature Extraction & Fine-Tuning) 학습 및 비교 분석 (RTX 5050 최적화)
- Feature Extraction: 백본 고정 + FC 헤드 학습
- Fine-Tuning: 차등 학습률(Differential Learning Rate) 적용
- Custom CNN vs ResNet Feature Extraction vs ResNet Fine-Tuning 종합 비교표 및 그래프 출력
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

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.stage1_data.dataset import get_dataloaders
from src.stage3_transfer_learning.model import build_resnet

MODELS_DIR = ROOT_DIR / "models"


def compute_confusion_matrix(y_true, y_pred, num_classes: int):
    """순수 NumPy 기반 Confusion Matrix 계산"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def evaluate(model, dataloader, criterion, device):
    """검증 루프"""
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
            
    return running_loss / total, 100.0 * correct / total, np.array(all_targets), np.array(all_preds)


def train_single_mode(
    mode: str = "feature_extraction",
    model_name: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 64,
    loaders: dict = None
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print(f"🚀 [3단계] ResNet ({model_name.upper()}) {mode.upper()} 학습 시작")
    print(f" - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" - Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 60)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    class_names = loaders["class_names"]

    model = build_resnet(model_name=model_name, num_classes=len(class_names), mode=mode, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저 구성 (Fine-Tuning 시 차등 학습률 적용)
    if mode == "feature_extraction":
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    else:  # fine_tuning
        backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
        fc_params = model.fc.parameters()
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-4},
            {"params": fc_params, "lr": 1e-3}
        ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = MODELS_DIR / f"{model_name}_{mode}_best.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
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
        
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "model_name": model_name,
                "mode": mode
            }, best_model_path)

    total_time = time.time() - start_time
    print(f"✅ {mode} 학습 완료 ({total_time:.1f}초)")

    # Test 평가
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    print(f"🎯 [{mode.upper()}] 최종 Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    
    return {
        "mode": mode,
        "history": history,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "y_true": y_true,
        "y_pred": y_pred,
        "path": best_model_path
    }


def run_transfer_learning_pipeline(model_name: str = "resnet18", epochs: int = 15, batch_size: int = 64):
    """Feature Extraction 및 Fine-Tuning 연속 실행 및 성능 비교 그래프 생성"""
    loaders = get_dataloaders(batch_size=batch_size)
    class_names = loaders["class_names"]
    
    # 1. Feature Extraction 실험
    fe_res = train_single_mode("feature_extraction", model_name, epochs=epochs, batch_size=batch_size, loaders=loaders)
    
    # 2. Fine-Tuning 실험
    ft_res = train_single_mode("fine_tuning", model_name, epochs=epochs, batch_size=batch_size, loaders=loaders)
    
    # 3. 종합 비교 그래프 작성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, epochs + 1)
    
    # Loss 비교
    ax1.plot(epochs_range, fe_res["history"]["val_loss"], 'b-o', label='Feature Extraction (Val Loss)')
    ax1.plot(epochs_range, ft_res["history"]["val_loss"], 'g--s', label='Fine-Tuning (Val Loss)')
    ax1.set_title('ResNet Transfer Learning: Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy 비교
    ax2.plot(epochs_range, fe_res["history"]["val_acc"], 'b-o', label=f'Feature Extraction (Test: {fe_res["test_acc"]:.1f}%)')
    ax2.plot(epochs_range, ft_res["history"]["val_acc"], 'g--s', label=f'Fine-Tuning (Test: {ft_res["test_acc"]:.1f}%)')
    ax2.set_title('ResNet Transfer Learning: Validation Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    comparison_path = MODELS_DIR / "resnet_transfer_comparison.png"
    plt.savefig(comparison_path, dpi=200)
    plt.close()
    print(f"\n📊 전이학습 비교 그래프 저장 완료: {comparison_path}")

    print("\n" + "=" * 60)
    print("🏆 최종 모델 성능 비교 요약")
    print(f"{'모델 / 학습 모드':<30} | {'Test Accuracy':<15} | {'Test Loss':<10}")
    print("-" * 60)
    print(f"{'ResNet18 Feature Extraction':<30} | {fe_res['test_acc']:>13.2f}% | {fe_res['test_loss']:>9.4f}")
    print(f"{'ResNet18 Fine-Tuning':<30} | {ft_res['test_acc']:>13.2f}% | {ft_res['test_loss']:>9.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet Transfer Learning Pipeline")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    
    run_transfer_learning_pipeline(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size)
