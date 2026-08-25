"""
1단계 (데이터 수집 및 전처리) 통합 실행 관리자 (CLI)
- 자동 크롤링 (클래스당 1,000장)
- 웹캠 대화형 캡처 툴 실행
- 데이터 정제 & pHash 중복 제거
- Train/Val/Test 데이터셋 분할
- PyTorch DataLoader 로딩 및 GPU 가속 검증
"""

import sys
import argparse
from pathlib import Path

# src 경로 추가
sys.path.append(str(Path(__file__).resolve().parent))

from src.stage1_data.collector import run_collection
from src.stage1_data.cleaner import clean_and_deduplicate
from src.stage1_data.splitter import split_dataset
from src.stage1_data.dataset import get_dataloaders, visualize_batch
import torch


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Footwear Dataset Collection & Preprocessing Pipeline")
    parser.add_argument("--all", action="store_true", help="전체 1단계 파이프라인 (크롤링 -> 정제 -> 분할 -> 로더 검증) 순차 실행")
    parser.add_argument("--crawl", action="store_true", help="이미지 크롤러 실행 (목표: 클래스당 1,000장)")
    parser.add_argument("--target-count", type=int, default=1000, help="클래스당 목표 수집 장수 (기본값: 1000)")
    parser.add_argument("--webcam", action="store_true", help="웹캠 대화형 캡처 툴 실행")
    parser.add_argument("--clean", action="store_true", help="데이터 정제 및 중복(pHash) 제거 실행")
    parser.add_argument("--split", action="store_true", help="Train/Val/Test 분할 (70:15:15) 실행")
    parser.add_argument("--test-loader", action="store_true", help="PyTorch DataLoader 및 RTX 5050 GPU 로딩 검증")
    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        print("🎯 1단계 전체 파이프라인 자동 실행을 시작합니다.")
        # 1. 크롤링
        run_collection(target_per_class=args.target_count)
        # 2. 정제
        clean_and_deduplicate()
        # 3. 분할
        split_dataset()
        # 4. 검증 및 시각화
        loaders = get_dataloaders(batch_size=64)
        visualize_batch(loaders["train"], loaders["class_names"])
        return

    if args.crawl:
        run_collection(target_per_class=args.target_count)

    if args.webcam:
        from src.stage1_data.webcam_capture import run_webcam_capture
        run_webcam_capture()

    if args.clean:
        clean_and_deduplicate()

    if args.split:
        split_dataset()

    if args.test_loader:
        print("🧪 PyTorch DataLoader 및 GPU 디바이스 검증...")
        print(f" - CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" - Active GPU: {torch.cuda.get_device_name(0)}")
        loaders = get_dataloaders(batch_size=64)
        
        # 첫 배치 로딩 시간 및 텐서 정보 측정
        images, labels = next(iter(loaders["train"]))
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            print(f"✅ GPU 텐서 로딩 성공: Images Shape = {images.shape}, Labels Shape = {labels.shape}, Device = {images.device}")
        else:
            print(f"✅ CPU 텐서 로딩 성공: Images Shape = {images.shape}, Labels Shape = {labels.shape}")
        
        visualize_batch(loaders["train"], loaders["class_names"])


if __name__ == "__main__":
    main()
