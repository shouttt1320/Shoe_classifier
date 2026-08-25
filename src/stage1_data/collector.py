"""
고속 병렬 멀티 엔진 이미지 크롤러 (icrawler Bing & Baidu 기반)
- 슬리퍼, 운동화, 크록스 3개 클래스를 멀티프로세스로 동시 병렬 수집
- 각 클래스당 1,000장 이상 목표 수집
- 손상 이미지 및 100x100 미만 저화질 필터링, RGB JPEG 자동 변환
"""

import os
import shutil
import hashlib
import tempfile
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from icrawler.builtin import BingImageCrawler, BaiduImageCrawler
from concurrent.futures import ProcessPoolExecutor

BASE_RAW_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/raw")

# 검색 키워드 정의 (다양한 각도, 브랜드, 색상, 형태, 언어)
KEYWORDS = {
    "slippers": [
        "slippers shoes", "indoor slippers", "house slippers", "slide sandals",
        "bathroom slippers", "flip flops", "fuzzy slippers", "men slippers",
        "women slippers", "hotel slippers", "warm slippers", "leather slippers",
        "슬리퍼", "삼선슬리퍼", "실내화", "거실화", "욕실슬리퍼", "여름슬리퍼", "쿠션슬리퍼"
    ],
    "sneakers": [
        "sneakers shoes", "running shoes", "sports sneakers", "athletic shoes",
        "canvas sneakers", "white sneakers", "tennis shoes", "men sneakers",
        "women sneakers", "basketball shoes", "casual sneakers", "running footwear",
        "운동화", "스니커즈", "런닝화", "단화", "헬스운동화", "워킹화", "트레이닝화"
    ],
    "crocs": [
        "crocs clogs", "crocs classic", "crocs shoes", "foam clogs",
        "crocs sandals", "crocs jibbitz", "platform crocs", "crocs slides",
        "white crocs", "black crocs", "color crocs", "garden clogs",
        "크록스", "크록스클로그", "크록스샌들", "크록스신발", "크록스슬리퍼", "EVA클로그"
    ]
}


def process_and_save_images(source_dir: Path, target_dir: Path, prefix: str, min_size: int = 100) -> int:
    """다운로드된 임시 이미지 파일들을 검증하고 유효한 이미지만 target_dir로 저장"""
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    
    for f in list(source_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            with Image.open(f) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    continue
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                file_hash = hashlib.md5((f.name + str(time.time())).encode()).hexdigest()[:8]
                dest_path = target_dir / f"{prefix}_{file_hash}_{f.name}"
                img.save(dest_path, "JPEG", quality=90)
                saved_count += 1
        except Exception:
            pass
        finally:
            try:
                f.unlink()
            except Exception:
                pass
                
    return saved_count


def crawl_category_worker(category: str, target_count: int = 1000):
    """단일 카테고리 수집 워커 (별도 프로세스로 실행)"""
    queries = KEYWORDS[category]
    cat_dir = BASE_RAW_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    current_count = len(list(cat_dir.glob("*.jpg")))
    print(f"[{category.upper()}] 시작 (기존: {current_count}/{target_count}장)")
    
    if current_count >= target_count:
        return category, current_count

    with tempfile.TemporaryDirectory(prefix=f"crawl_{category}_") as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. Bing Crawler
        for q in queries:
            current_count = len(list(cat_dir.glob("*.jpg")))
            if current_count >= target_count:
                break
                
            needed = target_count - current_count
            crawl_batch = min(120, max(50, needed + 15))
            
            try:
                crawler = BingImageCrawler(
                    downloader_threads=8,
                    storage={'root_dir': str(tmp_path)},
                    log_level=50  # CRITICAL only
                )
                crawler.crawl(keyword=q, max_num=crawl_batch)
                
                q_slug = q.replace(" ", "_")[:12]
                saved = process_and_save_images(tmp_path, cat_dir, prefix=f"bing_{q_slug}", min_size=100)
                current_count = len(list(cat_dir.glob("*.jpg")))
                print(f"  [{category.upper()}] '{q}' -> +{saved}장 저장 (누적: {current_count}/{target_count})")
                time.sleep(0.3)
            except Exception as e:
                time.sleep(0.5)

        # 2. 부족할 경우 Baidu Crawler
        current_count = len(list(cat_dir.glob("*.jpg")))
        if current_count < target_count:
            print(f"  [{category.upper()}] Bing 완료 후 부족분({target_count - current_count}장) Baidu 추가 수집...")
            for q in queries:
                current_count = len(list(cat_dir.glob("*.jpg")))
                if current_count >= target_count:
                    break
                try:
                    crawler = BaiduImageCrawler(
                        downloader_threads=8,
                        storage={'root_dir': str(tmp_path)},
                        log_level=50
                    )
                    crawler.crawl(keyword=q, max_num=80)
                    q_slug = q.replace(" ", "_")[:12]
                    saved = process_and_save_images(tmp_path, cat_dir, prefix=f"baidu_{q_slug}", min_size=100)
                    current_count = len(list(cat_dir.glob("*.jpg")))
                    print(f"  [{category.upper()}] Baidu '{q}' -> +{saved}장 저장 (누적: {current_count}/{target_count})")
                    time.sleep(0.3)
                except Exception:
                    pass

    final_count = len(list(cat_dir.glob("*.jpg")))
    print(f"✅ [{category.upper()}] 수집 완료: 총 {final_count}장")
    return category, final_count


def run_collection(target_per_class: int = 1000):
    """멀티프로세스를 이용한 3개 카테고리 동시 병렬 수집"""
    print("=" * 60)
    print(f"🚀 신발 3종 대용량 동시 병렬 데이터 수집 시작 (목표: 클래스당 {target_per_class}장)")
    print("=" * 60)

    categories = list(KEYWORDS.keys())
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(crawl_category_worker, cat, target_per_class) for cat in categories]
        results = [f.result() for f in futures]

    print("\n" + "=" * 60)
    print("🎉 모든 카테고리 수집 완료!")
    for cat, cnt in results:
        print(f" - {cat:<10}: {cnt}장")
    print("=" * 60)


if __name__ == "__main__":
    run_collection(target_per_class=1000)
