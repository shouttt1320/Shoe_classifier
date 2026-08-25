"""
데이터셋 분할 스크립트 (Train / Validation / Test)
- processed 디렉토리의 이미지를 70% : 15% : 15% 비율로 분할
- PyTorch ImageFolder 구조에 맞게 dataset/split/ 에 구성
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

BASE_PROCESSED_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/processed")
BASE_SPLIT_DIR = Path("/home/jw/Documents/Dapier/Test/dataset/split")
CATEGORIES = ["slippers", "sneakers", "crocs"]


def split_dataset(train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "비율의 합은 1.0이어야 합니다."
    
    random.seed(seed)
    print("=" * 60)
    print("✂️ 데이터셋 분할 (Train / Val / Test) 시작")
    print(f" - 분할 비율: Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%")
    print(f" - Random Seed: {seed}")
    print("=" * 60)

    # 기존 split 디렉토리 초기화
    if BASE_SPLIT_DIR.exists():
        shutil.rmtree(BASE_SPLIT_DIR)

    for split_name in ["train", "val", "test"]:
        for cat in CATEGORIES:
            (BASE_SPLIT_DIR / split_name / cat).mkdir(parents=True, exist_ok=True)

    summary = {cat: {"train": 0, "val": 0, "test": 0, "total": 0} for cat in CATEGORIES}

    for cat in CATEGORIES:
        proc_cat_dir = BASE_PROCESSED_DIR / cat
        if not proc_cat_dir.exists():
            print(f"⚠️ [{cat}] processed 디렉토리가 존재하지 않습니다.")
            continue

        images = sorted(list(proc_cat_dir.glob("*.jpg")) + list(proc_cat_dir.glob("*.png")))
        total_count = len(images)
        if total_count == 0:
            print(f"⚠️ [{cat}] 분할할 이미지가 없습니다.")
            continue

        # 셔플
        random.shuffle(images)

        n_train = int(total_count * train_ratio)
        n_val = int(total_count * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        splits = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }

        print(f"\n📁 [{cat.upper()}] 분할 복사 중... (총 {total_count}장)")
        for split_name, img_list in splits.items():
            dst_dir = BASE_SPLIT_DIR / split_name / cat
            for img_p in tqdm(img_list, desc=f"  -> {split_name:<5}", leave=False):
                shutil.copy2(img_p, dst_dir / img_p.name)
            summary[cat][split_name] = len(img_list)
        summary[cat]["total"] = total_count

    print("\n" + "=" * 60)
    print("📊 데이터셋 분할 완료 요약")
    print(f"{'클래스':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 60)
    total_train, total_val, total_test, grand_total = 0, 0, 0, 0
    for cat, s in summary.items():
        print(f"{cat:<12} | {s['train']:<8} | {s['val']:<8} | {s['test']:<8} | {s['total']:<8}")
        total_train += s['train']
        total_val += s['val']
        total_test += s['test']
        grand_total += s['total']
    print("-" * 60)
    print(f"{'합계':<12} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {grand_total:<8}")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset()
