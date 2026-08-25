"""
전체 프로젝트 (1단계 ~ 4단계) 통합 마스터 파이프라인 CLI
- 1단계: 데이터 수집, 웹캠 캡처, 정제, 분할, DataLoader 검증
- 2단계: Custom CNN 모델 구성 및 RTX 5050 가속 학습
- 3단계: ResNet 전이학습 (Feature Extraction & Fine-Tuning) 비교 분석
- 4단계: 실시간 웹캠 스트리밍 영상 추론
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트 및 src 디렉토리 등록
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import torch


def run_stage_1(target_count=1000):
    print("\n" + "=" * 70)
    print("📍 [STAGE 1] 데이터 수집, 정제 및 분할 파이프라인")
    print("=" * 70)
    from src.stage1_data.collector import run_collection
    from src.stage1_data.cleaner import clean_and_deduplicate
    from src.stage1_data.splitter import split_dataset
    from src.stage1_data.dataset import get_dataloaders, visualize_batch
    
    # 1. 크롤링
    run_collection(target_per_class=target_count)
    # 2. 정제
    clean_and_deduplicate()
    # 3. 분할
    split_dataset()
    # 4. 검증
    loaders = get_dataloaders(batch_size=64)
    visualize_batch(loaders["train"], loaders["class_names"])


def run_stage_2(epochs=25, batch_size=64, lr=1e-3):
    print("\n" + "=" * 70)
    print("📍 [STAGE 2] Custom CNN 아키텍처 학습 & 평가 (RTX 5050 가속)")
    print("=" * 70)
    from src.stage2_custom_cnn.train import train_custom_cnn
    train_custom_cnn(epochs=epochs, batch_size=batch_size, lr=lr)


def run_stage_3(model_name="resnet18", epochs=15, batch_size=64):
    print("\n" + "=" * 70)
    print("📍 [STAGE 3] ResNet 전이학습 (Feature Extraction & Fine-Tuning) 벤치마크")
    print("=" * 70)
    from src.stage3_transfer_learning.train import run_transfer_learning_pipeline
    run_transfer_learning_pipeline(model_name=model_name, epochs=epochs, batch_size=batch_size)


def run_stage_4(model_type="resnet", camera_id=0):
    print("\n" + "=" * 70)
    print("📍 [STAGE 4] 실시간 카메라 스트리밍 영상 추론")
    print("=" * 70)
    from src.stage4_camera_stream.stream_infer import run_streaming_inference
    run_streaming_inference(model_type=model_type, camera_id=camera_id)


def main():
    parser = argparse.ArgumentParser(description="Footwear 3-Class Classification Master Pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="실행할 단계 선택 (1, 2, 3, 4)")
    parser.add_argument("--all", action="store_true", help="1~3단계 전체 순차 자동 실행")
    parser.add_argument("--webcam-capture", action="store_true", help="웹캠 직접 촬영 툴 실행 (1단계 데이터 보강용)")
    parser.add_argument("--target-count", type=int, default=1000, help="1단계 클래스당 수집 목표치")
    parser.add_argument("--epochs", type=int, default=20, help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 사이즈")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스 (기본: 0)")
    args = parser.parse_args()

    if args.webcam_capture:
        from src.stage1_data.webcam_capture import run_webcam_capture
        run_webcam_capture(camera_id=args.camera)
        return

    if args.all:
        print("🚀 전체 파이프라인(1단계 -> 2단계 -> 3단계) 자동 실행을 시작합니다.")
        run_stage_1(target_count=args.target_count)
        run_stage_2(epochs=args.epochs, batch_size=args.batch_size)
        run_stage_3(epochs=args.epochs, batch_size=args.batch_size)
        print("\n✨ 1~3단계 학습 및 평가가 모두 완료되었습니다. 실시간 추론을 시작하려면 `python run_pipeline.py --stage 4`를 실행하세요.")
        return

    if args.stage == 1:
        run_stage_1(target_count=args.target_count)
    elif args.stage == 2:
        run_stage_2(epochs=args.epochs, batch_size=args.batch_size)
    elif args.stage == 3:
        run_stage_3(epochs=args.epochs, batch_size=args.batch_size)
    elif args.stage == 4:
        run_stage_4(camera_id=args.camera)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
