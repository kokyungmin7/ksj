# 개발 환경

## Python 패키지 매니저
- Python 환경 툴은 **uv**를 사용합니다.
- 스크립트 실행 시 `python` 대신 `uv run`을 사용하세요.
  - 예: `uv run main.py`, `uv run pytest`

## 실행 환경 분리
- **현재 코드 작성 환경**과 **VLM 학습/추론 환경**은 별개입니다.
- 모델 로드, 학습, 추론 관련 코드는 이 환경에서 직접 실행하지 마세요.
  - `Qwen3VLForConditionalGeneration`, `AutoProcessor`, DINOv3 관련 코드 실행 금지
  - 패키지 설치 확인, 문법 검사, 파일 읽기/쓰기 등 모델 무관한 작업만 이 환경에서 수행

## 추론 환경 스펙
- CUDA Version: 12.4
- Python: 3.10

---

# 프로젝트 설계 기록

## 개요
한국어 재활용 분류 VQA 데이터셋 (train.csv + 이미지 5,072개)에 대해
Qwen3-VL-8B-Instruct를 QLoRA로 파인튜닝하고, DINOv3로 이미지 ROI crop 및
개수 세기를 보조하는 파이프라인.

---

## 파일 구조

```
ksj/
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── default.yaml.example    # 기본 설정 템플릿 (git 관리)
│   ├── fast.yaml.example       # 빠른 실험용 템플릿 (git 관리)
│   ├── default.yaml            # 실제 설정 (gitignore)
│   ├── fast.yaml               # 실제 설정 (gitignore)
│   └── local.yaml              # 환경별 오버라이드 (gitignore)
│   └── local.yaml.example      # 오버라이드 템플릿 (git 관리)
├── data/
│   ├── __init__.py
│   └── dataset.py              # Dataset, DataCollator, train/val 분리
├── models/
│   ├── __init__.py
│   ├── qwen.py                 # Qwen3-VL 로드 + 추론 (원본/crop 2-image)
│   └── dino.py                 # DINOv3: attention crop + blob counting
├── pipeline/
│   ├── __init__.py
│   ├── router.py               # 개수 질문 키워드 판별
│   └── predictor.py            # 라우팅 + predict_with_trace()
├── training/
│   ├── __init__.py
│   └── trainer.py              # QLoRA SFTTrainer
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py            # val 평가 + 시각화 저장 + JSON 결과
├── utils/
│   ├── __init__.py
│   ├── prompt.py               # 프롬프트 빌더, 정답 추출
│   └── visualizer.py           # matplotlib 4-패널 시각화
└── main.py                     # CLI 진입점
```

---

## 전체 추론 흐름

```
이미지 + 질문 + 4지선다
        ↓
DINOv3 enabled?
  ├─ Yes →  attention map 추출 → ROI bbox crop
  │           ↓
  │         개수 질문? (키워드: 몇 개, 개수, 몇 가지 ...)
  │           ├─ Yes → dino_count (attention blob / reference similarity)
  │           │         → pick_answer_by_count → 정답
  │           │         (숫자 없으면 qwen_with_crop으로 fallback)
  │           └─ No  → qwen_predict_with_crop(원본 + crop) → 정답
  └─ No  →  qwen_predict(원본만) → 정답
        ↓
시각화 저장 (results/viz/{id}.png)
```

---

## 모델

| 모델 | 용도 | 로드 방식 |
|---|---|---|
| Qwen/Qwen3-VL-8B-Instruct | VQA 추론 / QLoRA 학습 | 4-bit NF4 (학습), bfloat16 (평가) |
| facebook/dinov3-vits16-pretrain-lvd1689m | attention crop + counting | float16, eager (output_attentions 필요) |

> **주의:** DINOv3는 `attn_implementation="eager"` 필수.
> `sdpa`는 `output_attentions=True`를 지원하지 않아 IndexError 발생.

---

## DINOv3 Counting 방식

### 기본: Attention Blob Counting
```
attention map (14×14) → bilinear upsample → threshold binarization
→ scipy.ndimage.label → blob 수 = 개수
```

### 확장: Reference-based Similarity Counting (`--reference-dir` 지정 시)
```
참조 이미지 patch features 평균 → 쿼리 이미지와 코사인 유사도 맵
→ threshold → blob 수 = 개수
(blob 0개면 attention 방식으로 fallback)
```

### 공통 Fallback
- 선택지에 숫자가 없으면 → Qwen3-VL로 fallback

---

## DINOv3 Attention Crop 방식

```
attention map → bilinear upsample (원본 크기)
→ threshold binary mask → ndimage.find_objects → bbox
→ crop_padding(20px) 추가 → crop
(crop < crop_min_size(64px)이면 원본 전체 사용)
```

모든 질문에 적용. crop은 Qwen3-VL 프롬프트에 두 번째 이미지로 전달.

---

## 프롬프트 형식 (Qwen3-VL)

```python
# content는 반드시 list 형식 (string이면 apply_chat_template에서 TypeError)
[
  {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
  {"role": "user",      "content": [
      {"type": "image", "image": "path/to/original"},
      {"type": "image", "image": "path/to/crop"},   # crop 있을 때만
      {"type": "text",  "text": "질문 + 선택지"},
  ]},
  {"role": "assistant", "content": [{"type": "text", "text": "b"}]},  # 학습 시만
]
```

---

## QLoRA 설정

| 항목 | 값 |
|---|---|
| 양자화 | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| target modules | q/k/v/o/gate/up/down_proj (LM layers) |
| Effective batch | 2 × grad_accum 8 = 16 |
| Epochs | 3 |
| LR | 2e-4 (cosine) |

---

## Train/Val 분리

```python
train_test_split(df, test_size=0.2, stratify=df["answer"], random_state=42)
# train: ~4,058  val: ~1,014
```
- `subset` 설정 시 stratified sampling 유지 (빠른 실험용)
- train/evaluate 모두 동일 seed → 항상 같은 val set

---

## Config 관리

- `default.yaml` / `fast.yaml` → **gitignore** (`.example`에서 복사해서 사용)
- `local.yaml` → **gitignore** (환경별 오버라이드, deep merge 적용)
- `main.py`의 `load_config()`가 자동으로 `local.yaml` 감지 후 merge

```bash
# 추론 환경 초기 셋업
cp configs/default.yaml.example configs/default.yaml
# 필요시 local.yaml 작성
cp configs/local.yaml.example configs/local.yaml
```

---

## 시각화 출력

`results/viz/{id}.png` — 4패널 (DINOv3 활성화 시):

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  원본+bbox   │ Attn Heatmap │ Attn Overlay │   ROI Crop   │
└──────────────┴──────────────┴──────────────┴──────────────┘
Q: 질문 텍스트
a) ...  b) ... ✓  c) ...  d) ...
경로: qwen_with_crop / dino_count(3개)  |  예측: b  |  정답: b  |  ✓ 정답
```

DINOv3 비활성화 시: 원본 이미지 1패널만 출력.

---

## 실행 명령

```bash
# 패키지 설치
uv sync

# zero-shot 평가
uv run main.py evaluate

# 빠른 실험 (500샘플)
uv run main.py train --config configs/fast.yaml

# 전체 학습
uv run main.py train

# fine-tuned 모델 평가
uv run main.py evaluate --checkpoint outputs/qwen3vl-lora

# reference 갤러리 지정
uv run main.py evaluate --reference-dir data/references/bong/

# DINOv3 비활성화 (local.yaml에 dino.enabled: false)
uv run main.py evaluate
```

---

## 알려진 이슈 및 수정 이력

| 이슈 | 원인 | 수정 |
|---|---|---|
| `IndexError: tuple index out of range` | DINOv3 `sdpa`는 `output_attentions` 미지원 | `load_dino()`에서 `attn_implementation="eager"` |
| `TypeError: string indices must be integers` | system/assistant content를 string으로 작성 | content를 `[{"type":"text","text":...}]` 형식으로 변경 |
| `pick_answer_by_count` 항상 None 반환 | `best_diff = best_key` 오타 | `best_diff = diff` 로 수정 |
