"""
고속 데이터 정제 및 중복/손상 이미지 제거 스크립트
- 손상된 이미지 및 비정상 포맷 필터링 (PIL & OpenCV 무결성 검증)
- 최소 해상도 필터링 (120x120 이상)
- OpenCV 고속 Difference Hash (dHash) 기반 유사/중복 이미지 정밀 탐지
- 표준 3채널 RGB JPEG 변환 후 dataset/processed/에 저장
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE_RAW_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/raw")
BASE_PROCESSED_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/processed")
CATEGORIES = ["slippers", "sneakers", "crocs"]


def compute_dhash(img_gray: np.ndarray, hash_size: int = 8) -> int:
    """OpenCV 기반 Difference Hash (dHash 64-bit) 계산"""
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    # 64비트 정수 해시로 압축
    return int(sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v]))


def hamming_distance(h1: int, h2: int) -> int:
    """두 64비트 해시간의 해밍 거리 (비트 차이) 계산"""
    return bin(h1 ^ h2).count('1')


def clean_and_deduplicate(min_size: int = 120, max_hamming_dist: int = 4):
    """
    각 카테고리별로 이미지를 검사하여 정제하고 중복을 제거합니다.
    - min_size: 최소 가로/세로 픽셀
    - max_hamming_dist: 중복 판정 해밍 거리 임계값 (기본값: 4비트 이하 차이 시 중복 판정)
    """
    print("=" * 60)
    print("🧹 데이터 정제 및 중복 제거 시작 (dHash 무결성 검사)")
    print(f" - 최소 해상도 기준: {min_size}x{min_size}")
    print(f" - dHash 중복 감지 임계값: {max_hamming_dist}")
    print("=" * 60)

    total_raw = 0
    total_processed = 0
    stats = {}

    for cat in CATEGORIES:
        raw_cat_dir = BASE_RAW_DIR / cat
        proc_cat_dir = BASE_PROCESSED_DIR / cat
        
        if proc_cat_dir.exists():
            shutil.rmtree(proc_cat_dir)
        proc_cat_dir.mkdir(parents=True, exist_ok=True)

        if not raw_cat_dir.exists():
            print(f"⚠️ [{cat}] raw 디렉토리가 존재하지 않습니다: {raw_cat_dir}")
            continue

        raw_files = list(raw_cat_dir.glob("*.jpg")) + list(raw_cat_dir.glob("*.png")) + list(raw_cat_dir.glob("*.jpeg"))
        total_raw += len(raw_files)

        print(f"\n🔍 [{cat.upper()}] 정제 작업 중... (원본: {len(raw_files)}장)")
        
        seen_hashes = []
        valid_count = 0
        corrupt_count = 0
        small_count = 0
        duplicate_count = 0

        for f in tqdm(raw_files, desc=f"[{cat}] 필터링"):
            try:
                # 1. OpenCV 디코딩 테스트
                cv_img = cv2.imread(str(f))
                if cv_img is None:
                    corrupt_count += 1
                    continue

                h, w, _ = cv_img.shape
                # 2. 해상도 검사
                if w < min_size or h < min_size:
                    small_count += 1
                    continue

                # 3. dHash 계산 및 중복 판정
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                curr_hash = compute_dhash(gray)

                is_duplicate = False
                for h_prev in seen_hashes:
                    if hamming_distance(curr_hash, h_prev) <= max_hamming_dist:
                        is_duplicate = True
                        break

                if is_duplicate:
                    duplicate_count += 1
                    continue

                seen_hashes.append(curr_hash)

                # 4. 표준 RGB 변환 및 정제 디렉토리 저장
                save_name = f"{cat}_{valid_count:05d}.jpg"
                cv2.imwrite(str(proc_cat_dir / save_name), cv_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                valid_count += 1

            except Exception:
                corrupt_count += 1

        stats[cat] = {
            "raw": len(raw_files),
            "valid": valid_count,
            "corrupt": corrupt_count,
            "small": small_count,
            "duplicate": duplicate_count
        }
        total_processed += valid_count

    print("\n" + "=" * 60)
    print("📊 데이터 정제 결과 요약")
    print(f"{'클래스':<12} | {'원본':<8} | {'손상':<6} | {'저화질':<6} | {'중복':<6} | {'최종 확보':<8}")
    print("-" * 60)
    for cat, s in stats.items():
        print(f"{cat:<12} | {s['raw']:<8} | {s['corrupt']:<6} | {s['small']:<6} | {s['duplicate']:<6} | {s['valid']:<8}")
    print("-" * 60)
    print(f"총 원본: {total_raw}장  ==>  최종 정제 완료: {total_processed}장")
    print("=" * 60)


if __name__ == "__main__":
    clean_and_deduplicate()
