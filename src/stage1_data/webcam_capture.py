"""
웹캠 대화형 신발 이미지 캡처 & 자동 라벨링 도구
- 슬리퍼(1), 운동화(2), 크록스(3) 카테고리 전환
- SPACE: 단일 컷 촬영 및 자동 저장
- B: 연사 모드 (다각도 연속 촬영 20장)
- G: 가이드 박스 표시 토글
- Q / ESC: 종료
"""

import os
import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime

# 저장 경로
BASE_RAW_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/raw")

CLASSES = {
    ord('1'): ("slippers", "슬리퍼 (Slippers)", (0, 165, 255)),   # 주황색
    ord('2'): ("sneakers", "운동화 (Sneakers)", (0, 255, 0)),     # 초록색
    ord('3'): ("crocs", "크록스 (Crocs)", (255, 0, 255))         # 보라색
}


def create_directories():
    """클래스별 raw 디렉토리 생성"""
    for cls_name, _, _ in CLASSES.values():
        (BASE_RAW_DIR / cls_name).mkdir(parents=True, exist_ok=True)


def get_class_counts():
    """디스크 내 클래스별 현재 이미지 수량 확인"""
    counts = {}
    for cls_name, _, _ in CLASSES.values():
        d = BASE_RAW_DIR / cls_name
        counts[cls_name] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return counts


def save_frame(frame, class_name: str, prefix: str = "webcam") -> Path:
    """프레임을 JPEG 이미지로 저장"""
    save_dir = BASE_RAW_DIR / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    file_path = save_dir / f"{prefix}_{class_name}_{timestamp}.jpg"
    cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return file_path


def run_webcam_capture(camera_id: int = 0, width: int = 1280, height: int = 720):
    create_directories()
    
    print("=" * 60)
    print("🎥 웹캠 신발 데이터 수집 도구 시작")
    print("  [1] 슬리퍼 선택   |  [2] 운동화 선택   |  [3] 크록스 선택")
    print("  [SPACE] 단발 촬영 |  [B] 연사 모드(20장) |  [G] 가이드 박스 토글")
    print("  [Q/ESC] 종료")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 카메라 {camera_id}번을 열 수 없습니다. 웹캠 연결을 확인해주세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    current_key = ord('1')
    current_class, current_label, current_color = CLASSES[current_key]
    
    show_guide = True
    burst_mode = False
    burst_remaining = 0
    burst_interval = 0.2  # 0.2초마다 1장
    last_burst_time = 0
    
    flash_effect = 0  # 캡처 시 흰색 플래시 효과 프레임 카운트
    session_captures = {c[0]: 0 for c in CLASSES.values()}

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽어올 수 없습니다.")
            break

        # FPS 계산
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        h, w, _ = frame.shape
        display_frame = frame.copy()

        # 연사 모드 처리
        if burst_mode and burst_remaining > 0:
            if curr_time - last_burst_time >= burst_interval:
                save_path = save_frame(frame, current_class, prefix="burst")
                session_captures[current_class] += 1
                burst_remaining -= 1
                last_burst_time = curr_time
                flash_effect = 2
                if burst_remaining == 0:
                    burst_mode = False

        # 플래시 효과
        if flash_effect > 0:
            flash_effect -= 1
            display_frame = cv2.addWeighted(display_frame, 0.4, 255 * (display_frame * 0 + 1).astype('uint8'), 0.6, 0)

        # 1. 가이드 박스 오버레이 (신발 위치 영역)
        if show_guide:
            box_w, box_h = int(w * 0.6), int(h * 0.65)
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), current_color, 2)
            cv2.putText(display_frame, "Fit Shoe in Box (Rotate for Diversity)", (x1 + 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

        # 2. 상단 HUD 정보 바
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)

        # 현재 선택된 클래스 표시
        cv2.putText(display_frame, f"CLASS: {current_label}", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, current_color, 2)

        # 디스크 및 세션 카운트
        counts = get_class_counts()
        info_text = f"Total: {counts[current_class]} imgs (Session +{session_captures[current_class]})"
        cv2.putText(display_frame, info_text, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 단축키 안내 및 FPS
        shortcut_text = "[1]Slip [2]Sneak [3]Croc | [SPACE]Shot [B]Burst(20) [G]Guide [Q]Quit"
        cv2.putText(display_frame, shortcut_text, (w - 680, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 연사 모드 진행 중 표시
        if burst_mode:
            burst_text = f"BURST RECORDING... {20 - burst_remaining}/20"
            cv2.putText(display_frame, burst_text, (w // 2 - 180, h - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Shoe Dataset Webcam Collector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key in CLASSES:
            current_key = key
            current_class, current_label, current_color = CLASSES[key]
        elif key == ord(' '):  # SPACE: 단발 촬영
            save_path = save_frame(frame, current_class, prefix="webcam")
            session_captures[current_class] += 1
            flash_effect = 3
            print(f"📸 캡처 완료 [{current_class}]: {save_path.name}")
        elif key == ord('b') or key == ord('B'):  # B: 연사 모드
            if not burst_mode:
                burst_mode = True
                burst_remaining = 20
                last_burst_time = 0
                print(f"🔥 연사 모드 시작 [{current_class}]: 20장 연속 저장 시작...")
        elif key == ord('g') or key == ord('G'):  # G: 가이드 토글
            show_guide = not show_guide

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 웹캠 데이터 수집기 종료.")
    print("최종 수집 현황:")
    for cls_name, label, _ in CLASSES.values():
        cnt = len(list((BASE_RAW_DIR / cls_name).glob("*.jpg")))
        print(f" - {label}: 총 {cnt}장 (금번 세션 +{session_captures[cls_name]}장)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam Shoe Image Capture Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    args = parser.parse_args()
    
    run_webcam_capture(camera_id=args.camera, width=args.width, height=args.height)
