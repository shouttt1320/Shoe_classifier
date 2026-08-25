# 👟 신발 3종(슬리퍼 / 운동화 / 크록스) 분류 딥러닝 프로젝트

> **NVIDIA RTX 5050 GPU 최적화 & Mixed Precision(AMP) 가속**  
> 1단계(데이터 수집/정제/웹캠 캡처)부터 2단계(Custom CNN), 3단계(ResNet 전이학습/파인튜닝), 4단계(실시간 웹캠 스트리밍 추론)까지 완성하는 엔드투엔드 파이프라인

---

## 📂 프로젝트 구조

```text
.
├── dataset/
│   ├── raw/                      # 원본 크롤링 및 웹캠 촬영 이미지
│   │   ├── slippers/             # 슬리퍼 이미지
│   │   ├── sneakers/             # 운동화 이미지
│   │   └── crocs/                # 크록스 이미지
│   ├── processed/                # 손상/저화질/pHash 중복 제거된 정제 데이터
│   └── split/                    # PyTorch ImageFolder 구조 (Train: 70%, Val: 15%, Test: 15%)
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── stage1_data/              # [1단계] 데이터 수집, 웹캠 촬영, 정제, 분할, DataLoader
│   │   ├── collector.py          # 클래스당 1,000장 이상 멀티엔진 자동 크롤러
│   │   ├── webcam_capture.py     # 웹캠 실시간 인터랙티브 캡처 & 자동 라벨링 툴
│   │   ├── cleaner.py            # 손상 이미지 및 pHash 유사 중복 제거기
│   │   ├── splitter.py           # Train / Val / Test 분할기
│   │   └── dataset.py            # RTX 5050 가속 PyTorch DataLoader 및 증강(Augmentation)
│   ├── stage2_custom_cnn/        # [2단계] Custom CNN 아키텍처 및 학습
│   ├── stage3_transfer_learning/ # [3단계] ResNet 전이학습 (Feature Extraction & Fine-Tuning)
│   └── stage4_camera_stream/     # [4단계] 실시간 웹캠 스트리밍 추론 & GUI
├── run_stage1.py                 # 1단계 통합 실행 CLI
├── requirements.txt              # 프로젝트 의존성 패키지
└── README.md
```

---

## ⚡ 1단계: 데이터 수집 및 전처리 사용법

### 1. 전체 파이프라인 원클릭 실행
```bash
# 클래스당 1,000장 크롤링 -> 정제 -> 분할 -> 로더 검증 일괄 실행
python3 run_stage1.py --all
```

### 2. 단계별 개별 실행

#### ① 웹 이미지 대량 크롤링 (클래스당 1,000장 목표)
```bash
python3 run_stage1.py --crawl --target-count 1000
```

#### ② 웹캠 실시간 인터랙티브 캡처 툴 (직접 신발 촬영)
```bash
python3 run_stage1.py --webcam
```
**단축키 안내:**
- `1`: **슬리퍼(Slippers)** 카테고리 선택
- `2`: **운동화(Sneakers)** 카테고리 선택
- `3`: **크록스(Crocs)** 카테고리 선택
- `SPACE`: 단일 컷 즉시 촬영 및 자동 라벨링 저장 (`dataset/raw/{category}/webcam_*.jpg`)
- `B`: **연사 모드 (Burst)** - 신발을 회전시키면서 0.2초 간격으로 20장 연속 자동 캡처
- `G`: 신발 위치 안내 가이드 박스 ON / OFF 토글
- `Q` 또는 `ESC`: 종료

#### ③ 데이터 정제 및 pHash 중복 제거
```bash
python3 run_stage1.py --clean
```
- 손상된 이미지 및 120x120 미만 저화질 이미지 자동 제외
- Perceptual Hash (pHash) 알고리즘으로 유사/중복 이미지를 정밀 탐지하여 제거
- 표준 3채널 RGB 포맷으로 `dataset/processed/`에 저장

#### ④ Train / Val / Test 데이터셋 분할 (70% : 15% : 15%)
```bash
python3 run_stage1.py --split
```

#### ⑤ PyTorch DataLoader 및 RTX 5050 GPU 가속 검증
```bash
python3 run_stage1.py --test-loader
```
- `batch_size=64`, `pin_memory=True`, `num_workers=4`의 고속 파이프라인 검증
- Data Augmentation(회전, 플립, 색상 변화, 정규화) 적용 샘플 그리드 이미지 (`dataset/sample_batch_grid.png`) 자동 생성

---

## 🚀 하드웨어 가속 (NVIDIA RTX 5050)
- **PyTorch 2.x + CUDA 13.0** 지원
- **AMP (Automatic Mixed Precision - FP16)** 적용으로 VRAM 절감 및 연산 가속
- 비동기 데이터 프리페칭(`pin_memory`, `persistent_workers`)을 통해 CPU-GPU 전송 병목 해소
