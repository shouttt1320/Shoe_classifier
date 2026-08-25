"""
Custom CNN 아키텍처 정의 (8-Layer Deep CNN, RTX 5050 최적화)
- Stage당 2개의 Conv2D를 연속 적용하는 DoubleConvBlock (총 8개 Conv 레이어)
- Receptive Field 확장 및 계층적 시각 특징 추출력 강화
- 채널 구조: 64 -> 128 -> 256 -> 512
- AdaptiveAvgPool2d((1, 1)) 및 Dropout(0.3) 분류기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """
    Double Conv 기본 블록 (Mish 활성화 함수 적용)
    Conv(3x3) -> BN -> Mish -> Conv(3x3) -> BN -> Mish -> MaxPool(2x2)
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            # 첫 번째 Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            # 두 번째 Conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.block(x))


class CustomFootwearCNN(nn.Module):
    """신발 3종 분류를 위한 8-Layer Deep Custom CNN (Mish Activation)"""
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        
        # 특징 추출기: 4개 Stage x 각 2개 Conv = 총 8개 Convolutional Layers
        self.features = nn.Sequential(
            DoubleConvBlock(3, 64, pool=True),    # Stage 1: 224x224 -> 112x112 (Conv 2개)
            DoubleConvBlock(64, 128, pool=True),  # Stage 2: 112x112 -> 56x56   (Conv 2개)
            DoubleConvBlock(128, 256, pool=True), # Stage 3: 56x56 -> 28x28     (Conv 2개)
            DoubleConvBlock(256, 512, pool=True)  # Stage 4: 28x28 -> 14x14     (Conv 2개)
        )
        
        # 적응형 풀링 (1x1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 분류기 (Classifier Head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat)
        out = self.classifier(pooled)
        return out


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFootwearCNN(num_classes=3).to(device)
    dummy_input = torch.randn(8, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print("=" * 60)
    print("🧠 8-Layer Deep CustomFootwearCNN 구조 요약")
    print(f" - 입력 텐서 Shape:  {dummy_input.shape}")
    print(f" - 출력 텐서 Shape:  {output.shape} (클래스 3개)")
    print(f" - 총 파라미터 수:   {count_parameters(model):,}개")
    print(f" - 사용 디바이스:    {device}")
    print("=" * 60)
