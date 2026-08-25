"""
ResNet 전이학습 모델 정의 (Feature Extraction & Fine-Tuning)
- torchvision Pretrained ResNet18 / ResNet50 백본 활용
- Feature Extraction: 백본 가중치 동결(Freeze) 후 최종 FC 헤드만 학습
- Fine-Tuning: 상위 레이어(Layer4) 또는 전체 계층 언프리즈(Unfreeze) 및 저학습률 미세조정
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet(
    model_name: str = "resnet18",
    num_classes: int = 3,
    mode: str = "feature_extraction",  # 'feature_extraction' 또는 'fine_tuning'
    pretrained: bool = True
) -> nn.Module:
    """
    ResNet 전이학습 모델 빌더
    """
    weights = "DEFAULT" if pretrained else None
    
    if model_name.lower() == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
    elif model_name.lower() == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}. ('resnet18', 'resnet50' 지원)")

    if mode == "feature_extraction":
        # 1. Feature Extraction: 모든 백본 레이어 동결
        for param in model.parameters():
            param.requires_grad = False
        print(f"🔒 [{model_name}] Feature Extraction 모드: 백본 가중치 동결 완료.")

    elif mode == "fine_tuning":
        # 2. Fine-Tuning: 기본적으로 전체 또는 상위 레이어(Layer 4) 학습 활성화
        for param in model.parameters():
            param.requires_grad = True
        print(f"🔓 [{model_name}] Fine-Tuning 모드: 전체 레이어 가중치 언프리즈 완료.")
        
    else:
        raise ValueError(f"알 수 없는 모드입니다: {mode}. ('feature_extraction', 'fine_tuning')")

    # 신발 3종 분류를 위한 새로운 FC 분류 헤드 구성
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_trainable_params(model: nn.Module):
    """학습 가능한 파라미터 수 및 전체 파라미터 수 반환"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Feature Extraction 테스트
    fe_model = build_resnet("resnet18", num_classes=3, mode="feature_extraction").to(device)
    fe_trainable, fe_total = get_trainable_params(fe_model)
    
    # 2. Fine-Tuning 테스트
    ft_model = build_resnet("resnet18", num_classes=3, mode="fine_tuning").to(device)
    ft_trainable, ft_total = get_trainable_params(ft_model)
    
    print("=" * 60)
    print("🧠 ResNet 전이학습 모델 비교")
    print(f" - Feature Extraction 학습 파라미터: {fe_trainable:,} / {fe_total:,} ({fe_trainable/fe_total*100:.2f}%)")
    print(f" - Fine-Tuning        학습 파라미터: {ft_trainable:,} / {ft_total:,} ({ft_trainable/ft_total*100:.2f}%)")
    print("=" * 60)
