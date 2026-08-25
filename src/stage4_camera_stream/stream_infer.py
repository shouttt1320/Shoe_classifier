"""
실시간 웹캠 영상 신발 3종 추론 GUI 애플리케이션 (RTX 5050 가속)
- 학습된 ResNet 전이학습 모델 또는 Custom CNN 모델 로드
- 실시간 GPU FP16 추론 및 Temporal Smoothing (확률 이동평균) 적용
- 클래스별 신뢰도(Confidence) 프로그레스 바, 실시간 FPS 및 지연시간(Latency) HUD 오버레이
"""

import os
import sys
import cv2
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.stage2_custom_cnn.model import CustomFootwearCNN
from src.stage3_transfer_learning.model import build_resnet

MODELS_DIR = ROOT_DIR / "models"

CLASS_LABELS_KO = {
    "slippers": "슬리퍼 (Slippers)",
    "sneakers": "운동화 (Sneakers)",
    "crocs": "크록스 (Crocs)"
}

CLASS_COLORS = {
    "slippers": (0, 165, 255),  # 주황
    "sneakers": (0, 255, 0),    # 초록
    "crocs": (255, 0, 255)      # 보라
}


def load_best_model(model_type: str = "resnet", weights_path: Path = None, device: str = "cuda"):
    """모델 로더"""
    if weights_path is None:
        if model_type == "resnet":
            weights_path = MODELS_DIR / "resnet18_fine_tuning_best.pth"
            if not weights_path.exists():
                weights_path = MODELS_DIR / "resnet18_feature_extraction_best.pth"
        else:
            weights_path = MODELS_DIR / "custom_cnn_best.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {weights_path}. 먼저 2단계 또는 3단계 학습을 진행해주세요.")

    checkpoint = torch.load(weights_path, map_location=device)
    class_names = checkpoint.get("class_names", ["crocs", "slippers", "sneakers"])
    
    if "resnet" in str(weights_path).lower() or model_type == "resnet":
        model = build_resnet(model_name="resnet18", num_classes=len(class_names), mode="fine_tuning", pretrained=False)
    else:
        model = CustomFootwearCNN(num_classes=len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ 모델 로드 완료: {weights_path.name} (Device: {device})")
    return model, class_names


def draw_hud(frame, probs, class_names, top_idx, fps, latency_ms, show_guide=True):
    """실시간 GUI 및 정보 오버레이 그리기"""
    h, w, _ = frame.shape
    top_class = class_names[top_idx]
    top_prob = probs[top_idx]
    accent_color = CLASS_COLORS.get(top_class, (0, 255, 0))

    # 1. 상단 타이틀 배너
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    
    # 2. 우측 하단 신뢰도 바 배경 박스
    panel_w, panel_h = 340, 160
    px1, py1 = w - panel_w - 20, h - panel_h - 20
    px2, py2 = w - 20, h - 20
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (25, 25, 25), -1)
    
    # 블렌딩 적용 (투명도)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 3. 상단 메인 감지 결과 텍스트
    pred_label = CLASS_LABELS_KO.get(top_class, top_class)
    cv2.putText(frame, f"DETECTED: {pred_label}", (25, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, accent_color, 2)
    cv2.putText(frame, f"Confidence: {top_prob * 100:.1f}%", (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # 4. 상단 우측 FPS & 지연시간
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 200, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (w - 200, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # 5. 가이드 박스
    if show_guide:
        box_w, box_h = int(w * 0.45), int(h * 0.55)
        gx1, gy1 = (w - box_w) // 2, (h - box_h) // 2
        gx2, gy2 = gx1 + box_w, gy1 + box_h
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), accent_color, 2)
        cv2.putText(frame, "Align Footwear Here", (gx1 + 10, gy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, accent_color, 2)

    # 6. 클래스별 확률 프로그레스 바 (Confidence Bars)
    cv2.putText(frame, "CLASS PROBABILITIES", (px1 + 15, py1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    bar_max_w = 170
    for i, cls in enumerate(class_names):
        p = probs[i]
        c_name_kr = CLASS_LABELS_KO.get(cls, cls).split(" ")[0]
        col = CLASS_COLORS.get(cls, (200, 200, 200))
        
        y_offset = py1 + 55 + i * 32
        # 라벨
        cv2.putText(frame, f"{c_name_kr:<5}", (px1 + 15, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        
        # 바 배경
        cv2.rectangle(frame, (px1 + 80, y_offset), (px1 + 80 + bar_max_w, y_offset + 15), (50, 50, 50), -1)
        # 바 채우기
        curr_bar_w = int(bar_max_w * p)
        if curr_bar_w > 0:
            cv2.rectangle(frame, (px1 + 80, y_offset), (px1 + 80 + curr_bar_w, y_offset + 15), col, -1)
        
        # 수치 표시
        cv2.putText(frame, f"{p*100:4.1f}%", (px1 + 80 + bar_max_w + 10, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 단축키 안내
    cv2.putText(frame, "[G] Guide Toggle | [Q/ESC] Quit", (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def run_streaming_inference(model_type: str = "resnet", camera_id: int = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_best_model(model_type=model_type, device=device)

    # 전처리 파이프라인
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 웹캠 {camera_id}번을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print("🎥 실시간 신발 3종 추론 스트리밍 시작")
    print(" - [G]: 가이드 박스 표시 토글")
    print(" - [Q / ESC]: 종료")
    print("=" * 60)

    smoothed_probs = np.zeros(len(class_names))
    alpha = 0.25  # Exponential moving average 계수 (부드러운 전이)

    fps = 0
    prev_time = time.time()
    show_guide = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        # 전처리 & GPU 추론
        t0 = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        latency_ms = (time.time() - t0) * 1000.0

        # Temporal Smoothing
        smoothed_probs = (1 - alpha) * smoothed_probs + alpha * probs
        top_idx = int(np.argmax(smoothed_probs))

        # HUD 렌더링
        rendered_frame = draw_hud(frame, smoothed_probs, class_names, top_idx, fps, latency_ms, show_guide=show_guide)

        cv2.imshow("Footwear AI Classifier - RTX 5050 Stream", rendered_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('g') or key == ord('G'):
            show_guide = not show_guide

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Footwear Camera Streaming Inference")
    parser.add_argument("--model-type", type=str, default="resnet", choices=["resnet", "custom_cnn"])
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    run_streaming_inference(model_type=args.model_type, camera_id=args.camera)
