"""
PyTorch Dataset & DataLoader 구성 (RTX 5050 최적화)
- ImageNet 표준 정규화 및 Data Augmentation 적용
- GPU(RTX 5050) 가속용 pin_memory, multi-worker 지원
- 배치 데이터 시각화 및 검증 기능 포함
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_SPLIT_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/split")

# ImageNet 표준 통계치
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    """학습용 증강(Augmentation) 및 검증용 변환 정의"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_test_transform


def get_dataloaders(
    data_dir: Path = BASE_SPLIT_DIR,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Train, Val, Test DataLoader 생성 (RTX 5050 최적화)
    """
    train_tf, val_test_tf = get_transforms(img_size=img_size)

    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"데이터셋 디렉토리를 찾을 수 없습니다: {data_dir}. 먼저 splitter.py를 실행해주세요.")

    train_dataset = datasets.ImageFolder(root=str(train_path), transform=train_tf)
    val_dataset = datasets.ImageFolder(root=str(val_path), transform=val_test_tf)
    test_dataset = datasets.ImageFolder(root=str(test_path), transform=val_test_tf)

    # CUDA 사용 여부에 따른 설정 조정
    cuda_available = torch.cuda.is_available()
    pin_memory = pin_memory and cuda_available
    
    # worker 설정 (리눅스 환경 최적화)
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    class_names = train_dataset.classes
    print(f"📦 DataLoader 준비 완료:")
    print(f" - 클래스: {class_names} (총 {len(class_names)}개)")
    print(f" - Train 샘플 수: {len(train_dataset)}장 ({len(train_loader)} 배달 배치)")
    print(f" - Val 샘플 수:   {len(val_dataset)}장 ({len(val_loader)} 배달 배치)")
    print(f" - Test 샘플 수:  {len(test_dataset)}장 ({len(test_loader)} 배달 배치)")
    print(f" - Batch Size: {batch_size}, Pin Memory: {pin_memory}, Num Workers: {num_workers}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": class_names,
        "datasets": {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
    }


def denormalize(tensor):
    """시각화를 위해 정규화된 텐서를 [0, 1] 범위 넘파이 이미지로 역변환"""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def visualize_batch(dataloader, class_names, save_path: str = "/home/jw/Documents/Dapier/Test/dataset/sample_batch_grid.png", max_imgs: int = 16):
    """DataLoader에서 첫 배치를 가져와 증강된 샘플 그리드 시각화 저장"""
    images, labels = next(iter(dataloader))
    num_display = min(len(images), max_imgs)
    
    rows = int(np.ceil(np.sqrt(num_display)))
    cols = int(np.ceil(num_display / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i in range(num_display):
        img = denormalize(images[i])
        label_idx = labels[i].item()
        cls_name = class_names[label_idx]
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {cls_name}", fontsize=12, fontweight='bold')
        axes[i].axis('off')

    for j in range(num_display, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🖼️ 샘플 배치 시각화 이미지 저장 완료: {save_path}")


if __name__ == "__main__":
    try:
        loaders = get_dataloaders(batch_size=16)
        visualize_batch(loaders["train"], loaders["class_names"])
    except Exception as e:
        print(f"⚠️ 테스트 실패: {e}")
