# 🤖 학습된 모델 파일

## ⚠️ 다운로드 안내

모델 파일은 용량이 커서(~400MB) GitHub에 직접 업로드하지 않았습니다.
모델 필요시 메일로 연락 부탁드립니다. xogur1578@gmail.com



---

## 📦 모델 목록

### Phase 1: 기본 텍스트 분류 모델

| 파일명 | 레이블 | 텍스트 범위 | 모델 | 크기 |
|--------|--------|-------------|------|------|
| `phase1_six_label_all_text_mlp.pt` | 6-class | 전체 대화 | MLP | ~50MB |
| `phase1_six_label_all_text_finetuning.pt` | 6-class | 전체 대화 | Fine-tuning | ~430MB |
| `phase1_six_label_first_text_mlp.pt` | 6-class | 첫 문장 | MLP | ~50MB |
| `phase1_six_label_first_text_finetuning.pt` | 6-class | 첫 문장 | Fine-tuning | ~430MB |
| `phase1_four_label_all_text_mlp.pt` | 4-class | 전체 대화 | MLP | ~50MB |
| `phase1_four_label_all_text_finetuning.pt` | 4-class | 전체 대화 | Fine-tuning | ~430MB |
| `phase1_four_label_first_text_mlp.pt` | 4-class | 첫 문장 | MLP | ~50MB |
| `phase1_four_label_first_text_finetuning.pt` | 4-class | 첫 문장 | Fine-tuning | ~430MB |

**레이블 설명:**
- **6-class**: 기쁨, 당황, 분노, 불안, 상처, 슬픔
- **4-class**: 일반(기쁨+당황+분노), 우울(불안+상처+슬픔)

---

### Phase 2: PHQ-9 특징 통합 모델

| 파일명 | 레이블 | 특징 | 크기 |
|--------|--------|------|------|
| `phase2_six_label_all_text_phq9.pt` | 6-class | Text + PHQ-9 유사도 | ~430MB |
| `phase2_four_label_all_text_phq9.pt` | 4-class | Text + PHQ-9 유사도 | ~430MB |

**주요 개선:**
- PHQ-9 우울증 척도 키워드 유사도 추가
- 키워드 매칭 → 의미 유사도 방식 전환

---

### Phase 3: 멀티모달 융합 모델 ⭐ **최종 모델**

| 파일명 | 설명 | 크기 |
|--------|------|------|
| `phase3_six_label_all_text_phq9.pt` | Text + PHQ-9 (Fine-tuning 단독) | ~430MB |
| `phase3_six_label_all_text_phq9_multimodal.pt` | **Text + Audio + PHQ-9 (멀티모달)** | ~450MB |

**최종 성능:**
- Fine-tuning 단독: Accuracy 0.711, Depression Recall 0.902
- Multimodal: Accuracy 0.647, Depression Recall 0.827

---

## 🚀 사용 방법

### 1. 모델 다운로드

필요한 모델 다운로드 후 이 폴더에 저장:

```
models/
├── README.md
└── phase3_six_label_all_text_phq9_multimodal.pt  ← 여기에 저장
```

### 2. 모델 로드 (Python)

```python
from src.model import load_model

# 멀티모달 모델 로드
model = load_model(
    checkpoint_path='models/phase3_six_label_all_text_phq9_multimodal.pt',
    num_class=6,
    device='cuda'
)
```

### 3. 추론 실행

```python
from src.inference import DepressionDetector

detector = DepressionDetector(
    model_path='models/phase3_six_label_all_text_phq9_multimodal.pt',
    device='cuda'
)

result = detector.predict(
    text="요즘 너무 우울하고 힘들어요",
    audio_path="data/sample.mp3"
)
```

---

## 📊 모델 선택 가이드

### 용도별 추천 모델

| 용도 | 추천 모델 | 이유 |
|------|----------|------|
| **실제 배포용** | `phase3_six_label_all_text_phq9.pt` | 텍스트만 필요, 높은 Recall |
| **연구/실험용** | `phase3_six_label_all_text_phq9_multimodal.pt` | 멀티모달 결합 확인 |
| **빠른 프로토타입** | `phase1_six_label_all_text_mlp.pt` | 작은 용량, 빠른 추론 |
| **높은 정확도** | `phase1_six_label_all_text_finetuning.pt` | 기본 성능 검증 |

### 우울 신호 감지 최적화

**False Negative 최소화가 목표**라면:
→ `phase3_six_label_all_text_phq9.pt` (Depression Recall 0.902)

**멀티모달 실험**이 목표라면:
→ `phase3_six_label_all_text_phq9_multimodal.pt`

---

## 🔧 모델 상세 정보

### Phase 3 Multimodal 모델 구조

```
입력:
├── 텍스트 → KLUE RoBERTa (부분 Fine-tuning)
├── PHQ-9 유사도 → Linear Layer (16-dim)
└── 음성(MFCC) → BiLSTM

결합: Late Fusion (평균)

출력: 6-class 확률 분포
```

### 학습 설정

```python
# 하이퍼파라미터
batch_size = 8
epochs = 6
learning_rate = 1e-5 (text), 1e-4 (classifier)
optimizer = AdamW
weight_decay = 0.01
dropout = 0.3

# 데이터
train_size = 2,013
test_size = 863
class_balancing = WeightedRandomSampler
```

---

## ⚙️ .gitignore 설정

`.pt` 파일은 용량이 커서 Git에 추적하지 않습니다.

```gitignore
# .gitignore
models/*.pt
models/*.pth
```

모델 파일은 Google Drive 또는 AWS S3에 업로드하고 링크로 공유하세요.

---

## 📝 주의사항

1. **용량 문제**: GitHub 파일 크기 제한 100MB → 모델은 외부 저장소 사용 필수
2. **버전 관리**: 모델 업데이트 시 파일명에 날짜 또는 버전 추가 권장
3. **보안**: 개인정보가 포함된 데이터로 학습한 모델은 공개 금지

---

## 🔗 관련 링크

- [메인 README](../README.md)
- [코드 사용법](../src/README.md)
- [실험 노트북](../notebooks/)
- [성능 비교 결과](../results/)

---

**마지막 업데이트**: 2024.11.24  
**총 모델 수**: 12개  
**최종 모델**: `phase3_six_label_all_text_phq9_multimodal.pt`
