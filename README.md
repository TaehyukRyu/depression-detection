
# 멀티모달 기반 청소년 정서 위험 조기 인식 및 분류 시스템

<div align="center">


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)


</div>

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [데이터셋](#-데이터셋)
- [실험 결과](#-실험-결과)
- [설치 및 실행](#-설치-및-실행)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [제한사항 및 향후 과제](#-제한사항-및-향후-과제)

---

## 🎯 프로젝트 개요

### 배경 및 필요성

기존 감성 분석 모델은 일상적인 부정 감정(짜증, 화남)과 임상적 주의가 필요한 우울 신호를 명확히 구분하지 못하는 한계가 있습니다. 특히 청소년의 대화에서는 단순한 불만과 실제 위험 신호가 혼재되어 나타나므로, 이를 정밀하게 구분할 수 있는 기술이 필요합니다.

### 프로젝트 목표

- **우울 신호 탐지율(Recall) 극대화**: False Negative를 최소화하여 실제 위험군을 놓치지 않는 시스템 구축
- **우울 신호 그룹화**: 불안/상처/슬픔을 "우울 신호"로 재분류
- **전체 문맥 활용**: 3턴 대화 전체를 입력으로 사용
- **PHQ-9 통합**: 우울증 선별검사 기준을 특징으로 추가
- **멀티모달 융합**: 텍스트 의미 + 음성 운율 결합

### 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **기간** | 2025.10 - 2025.11 |
| **배포 URL** | http://43.201.10.85:8501 |
| **실험 구성** | 3 Phases, 14개 체계적 비교 실험 |

---

## 🌟 주요 특징

### 1. 체계적인 3단계 실험 설계

각 Phase마다 단 하나의 변수만 변경하여 그 효과를 명확히 측정했습니다.

| Phase | 목표 | 실험 수 | 핵심 결과 |
|-------|------|---------|----------|
| **Phase 1** | 라벨링 & 문맥 비교 | 8개 | 6-class + 전체 문맥 + Fine-tuning 선정 |
| **Phase 2** | PHQ-9 효과 검증 | 4개 | Recall +1.17%p 향상 |
| **Phase 3** | 멀티모달 융합 | 2개 | 성능 하락 → 원인 분석 |

**재현 가능성 보장**: 모든 실험에서 Random Seed(42) 고정, Train/Test 비율(80:20) 유지

### 2. 우울 신호 정의 및 라벨링 전략

AI Hub 감성 대화 말뭉치의 6개 감정 대분류 중 불안(3), 상처(4), 슬픔(5)을 "우울 신호"로 그룹화

| 라벨 | 대분류 | 주요 소분류 감정 | 분류 |
|------|--------|-----------------|------|
| 0 | 기쁨 | 감사, 신뢰, 편안, 만족, 흥분 | 비우울 |
| 1 | 당황 | 고립된(당황), 외로운, 열등감, 부끄러운 | 비우울 |
| 2 | 분노 | 툴툴대는, 짜증내는, 노여워하는, 성가신 | 비우울 |
| 3 | 불안 | 두려운, 스트레스받는, 취약한, 걱정스러운 | ⚠️ 우울 신호 |
| 4 | 상처 | 고립된, 배신당한, 버려진, 충격받은, 괴로워하는 | ⚠️ 우울 신호 |
| 5 | 슬픔 | 우울한, 좌절한, 비통한, 염세적인, 낙담한 | ⚠️ 우울 신호 |

### 3. PHQ-9 도메인 지식 주입

**PHQ-9**: DSM-5의 주요 우울증 진단 기준 9가지를 반영한 자가보고식 선별 검사지

**방법론**: PHQ-9 키워드 벡터와 입력 텍스트 벡터 간의 코사인 유사도 계산

```python
similarity = cosine_similarity(text_emb, phq9_emb)

# 3단계 가중치 적용
Direct Core (x3.0): 자살/자해 직접 표현
Core List (x2.0): 우울/무기력/집중력 저하
Indirect List (x1.0): 수면/식욕 변화
```

**중요한 실험적 발견**: 키워드 매칭은 오히려 노이즈가 되었습니다. 최종적으로 '유사도'만 사용한 모델이 더 높은 성능을 보여 도메인 지식의 '맥락적' 적용이 중요함을 확인했습니다.

### 4. 멀티모달 융합 아키텍처

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    PHQ-9        │     │   Text Input    │     │   Audio Input   │
│ (similarity)    │     │                 │     │  (MFCC 30-dim)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Linear Layer   │     │  KLUE RoBERTa   │     │      LSTM       │
│  1-dim → 16-dim │     │ (Partial Fine-  │     │ Hidden: 128     │
│     + ReLU      │     │  tuning Top 3)  │     │ Layers: 2       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐     ┌─────────────────┐
         │              │ CLS Token Pool  │     │ Adaptive Pool   │
         │              │    (768-dim)    │     │   (128-dim)     │
         │              └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────┬───────────┘                       │
                     ▼                                   │
            ┌─────────────────┐                          │
            │   Concatenate   │                          │
            │ CLS(768)+PHQ(16)│                          │
            └────────┬────────┘                          │
                     ▼                                   ▼
            ┌─────────────────┐                 ┌─────────────────┐
            │ Text Classifier │                 │Audio Classifier │
            │ Linear(784→6)   │                 │ Linear(128→6)   │
            └────────┬────────┘                 └────────┬────────┘
                     │                                   │
                     └─────────────┬─────────────────────┘
                                   ▼
                          ┌─────────────────┐
                          │   Late Fusion   │
                          │ (Text + Audio)/2│
                          └─────────────────┘
```

---

## 📊 데이터셋

### 텍스트 데이터
- **출처**: AI Hub 감성 대화 말뭉치 (청소년 필터링)
- **규모**: 10,582건
- **구성**: 발화 텍스트 (3개 턴 대화 연결)
- **Tokenizer**: KLUE/RoBERTa-base
- **Max Length**: 71 tokens (90% Percentile 기준)

### 음성 데이터
- **규모**: 2,876건 (멀티모달 학습용)
- **형식**: MP3
- **전처리**:
  - Sample Rate: 16,000Hz
  - MFCC: 30 coefficients
  - Trim Silence: top_db=20
  - RMS 정규화: target_rms=0.05

### 클래스 분포 (총 10,582건)

```
불안: 1,998건 | 슬픔: 1,887건 | 당황: 1,844건
상처: 1,818건 | 분노: 1,792건 | 기쁨: 1,243건

→ 우울 신호 (불안+상처+슬픔): 5,703건 (53.9%)
→ 비우울 신호 (기쁨+당황+분노): 4,879건 (46.1%)
```

---

## 📈 실험 결과

### Phase 1: 라벨링 & 문맥 비교 (8개 모델)

| Task | Text | Label | Model | Accuracy | Depression Recall |
|------|------|-------|-------|----------|-------------------|
| six_label_all_text | all | 6-class | MLP | 49.32% | 48.45% |
| **six_label_all_text** | **all** | **6-class** | **Fine-tuning** | **69.07%** | **82.29%** |
| four_label_all_text | all | 4-class | MLP | 63.56% | 32.92% |
| four_label_all_text | all | 4-class | Fine-tuning | 72.88% | 73.62% |

**핵심 발견**:
1. **6-class 우수**: "짜증" vs "좌절감" 같은 미묘한 차이가 중요 단서
2. **전체 문맥 필수**: 대화 흐름 파악으로 첫 문장 대비 Recall 1.17%p 상승
3. **Fine-tuning 압도적**: MLP 방식(48.45%) 대비 33.84%p 높은 성능

---

### Phase 2: PHQ-9 효과 검증 (4개 모델)

| Task | PHQ-9 | Accuracy | Depression Recall | False Negatives |
|------|-------|----------|-------------------|-----------------|
| Baseline (PHQ-9 없음) | ❌ | 69.07% | 82.29% | 303건 (17.7%) |
| **6-class + PHQ-9** | **✅** | **69.23%** | **83.46%** | **283건 (16.5%)** |

**핵심 인사이트**:
1. **도메인 지식 효과**: 임상 심리학 진단 기준 주입으로 유의미한 성능 향상
2. **고효율**: 추가 데이터 학습 없이 유사도 특징(1차원)만으로 우울 신호 20건 추가 탐지
3. **자살/자해 탐지**: 임상적으로 중요한 고위험 표현 탐지 강화에 기여

---

### Phase 3: 멀티모달 융합 (2개 모델)

| Task | Modality | Accuracy | Depression Precision | Depression Recall |
|------|----------|----------|----------------------|-------------------|
| **텍스트 단독 (Best)** | 📝 | **71.15%** | 94.59% | **90.16%** |
| 멀티모달 융합 | 📝+🎤 | 64.66% | **97.42%** | 82.66% |

**⚠️ 성능 하락**: Recall -7.5%p, False Negative +48건 증가

**성능 하락 원인 분석**:

1. **모달리티 간 성능 불균형 (핵심 원인)**
   - 텍스트 모델(사전학습+Fine-tuning)은 고성능
   - 음성 모델(LSTM)은 상대적 저성능
   - 융합 시 저성능 음성 모델이 텍스트 모델의 판단력을 희석

2. **컴퓨팅 리소스 제약**
   - Colab 무료버전 한계로 Wav2Vec 2.0 같은 고성능 음성 모델 실험 불가

3. **음성 데이터 품질 문제**
   - 연기된 데이터셋 특성상 실제 우울 환자의 미묘한 운율적 특징 부족

---

### 최종 성능 요약

```
✨ 최종 배포 모델: Multimodal (Text + Audio + PHQ-9) + 6-class
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Overall Accuracy:          64.66%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 우울 신호 탐지 성능 (불안+상처+슬픔):
   - Precision:  97.42%  (False Positive 최소화) ⭐
   - Recall:     82.66%  (실제 위험군 포착)
   - F1-Score:   89.43%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ False Negative: 111/640건 (17.3%)
✅ True Positive:  529/640건 (82.7%)
```

**배포 결정 이유**: 현재는 텍스트 단독 모델(71.15%)이 우세하나, 운율·톤·발화 속도 등 음성의 비언어적 정보 활용을 위해 멀티모달 모델을 배포. 음성 특징 추출 기법 개선 시 성능 향상 가능성이 높음.

---

## 🚀 설치 및 실행

### 방법 1: Docker Compose (권장)

```bash
# 레포지토리 클론
git clone https://github.com/TaehyukRyu/depression-detection.git
cd depression-detection

# 모델 파일 다운로드 (용량 문제로 별도 제공)
# models/ 폴더에 phase3_six_label_all_text_phq9_multimodal.pt 배치

# Docker Compose 실행
docker-compose up --build
```

실행 후 접속:
- **Frontend (Streamlit)**: http://localhost:8501
- **Backend (FastAPI)**: http://localhost:8000

### 방법 2: 로컬 실행

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 백엔드 의존성 설치 및 실행
cd src/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 새 터미널에서 프론트엔드 실행
cd src/frontend
pip install -r requirements.txt
streamlit run app.py
```

### 추론 예시 (Python)

```python
from inference import DepressionDetector

detector = DepressionDetector(
    model_path='models/phase3_six_label_all_text_phq9_multimodal.pt'
)

result = detector.predict(
    text="요즘 아무것도 하기 싫고 자꾸 힘든 생각이 들어",
    audio_path="sample.wav"
)

print(f"예측 감정: {result['label']}")
print(f"우울 신호 여부: {result['is_depression']}")
print(f"신뢰도: {result['confidence']:.2%}")
```

---

## 🛠 기술 스택

### AI/ML

| 기술 | 용도 | 버전 |
|------|------|------|
| PyTorch | 딥러닝 프레임워크 | 2.0+ |
| Transformers | KLUE-RoBERTa 모델 | 4.30+ |
| Librosa | 음성 특징 추출 (MFCC) | 0.10.0 |
| scikit-learn | 데이터 전처리 및 평가 | - |

### Backend/Deployment

| 기술 | 용도 |
|------|------|
| FastAPI | REST API 서버 |
| Docker | 컨테이너화 |
| AWS EC2 | 클라우드 배포 |

### Frontend

| 기술 | 용도 |
|------|------|
| Streamlit | 웹 인터페이스 |

---

## 📁 프로젝트 구조

```
depression-detection/
│
├── docs/                                    # 📚 문서 및 연구 자료
│   ├── 018_감성대화_데이터_구축_가이드라인.pdf
│   ├── 06_[자연어영역]_감성_대화_말뭉치.pdf
│   ├── AI_Hub_데이터참고.pdf
│   ├── Phase_1_라벨_문맥_모델_비교.pdf
│   ├── Phase_2_PHQ9_가중치_효과.pdf
│   ├── Phase_3_멀티모달_최종.pdf
│   ├── 음성데이터_전처리.pdf
│   ├── 청소년_우울_신호_탐지_시스템_포트폴리오.pdf
│   └── 프로젝트_계획서.docx
│
├── models/                                  # 🤖 학습된 모델 파일(대용량 업로드 X)
│   └── models_README.md                     # 모델 다운로드 안내
│
├── notebooks/                               # 📓 실험 Jupyter 노트북
│   ├── Phase_1_라벨_문맥_모델_비교.ipynb
│   ├── Phase_2_PHQ9_가중치_효과.ipynb
│   ├── Phase_3_멀티모달_최종.ipynb
│   └── 음성데이터_전처리.ipynb
│
├── results/                                 # 📊 실험 결과 CSV
│   ├── phase1_total_result.csv              # 8개 모델 비교
│   ├── phase2_total_result.csv              # PHQ-9 효과 비교
│   └── phase3_total_result.csv              # 멀티모달 비교
│
├── src/                                     # 💻 소스 코드
│   ├── backend/                             # Fast API 백엔드
│   │   ├── Dockerfile
│   │   ├── main.py                          
│   │   ├── model.py                         
│   │   ├── inference.py                     
│   │   ├── preprocessing.py                 
│   │   └── requirements.txt
│   │
│   └── frontend/                            # Streamlit 프론트엔드
│   │    ├── Dockerfile
│   │    ├── app.py                           # 웹 인터페이스
│   │    └── requirements.txt
│   │
│   └── docker-compose.yml
│
│
└── README.md                                
```

---

## 📌 제한사항 및 향후 과제

### 현재 제한사항

**1. 멀티모달 성능 불균형 문제** ⚠️
- **텍스트 모델**: KLUE-RoBERTa 전체 파라미터 Fine-tuning 수행
- **음성 모델**: 2,876건으로 LSTM 기반 학습 (파인튜닝 미수행)
- **문제**: 
  - 텍스트와 음성의 학습 규모 차이
  - 텍스트는 사전학습 모델 활용, 음성은 LSTM 학습
  - 두 모달리티의 특징을 평균/결합하면서 고성능 텍스트 모델이 저성능 음성 모델에 의해 희석됨
- **결과**: 멀티모달 Accuracy 64.66% (Text Only 71.15% 대비 6.49%p 하락)

**2. 컴퓨팅 리소스 제약** ⚠️
- **GPU 부족**: Google Colab 무료 버전의 제한된 GPU 사용 시간
  - 멀티모달 전체 학습 불가 (메모리 부족으로 배치 사이즈 축소)
  - 음성 사전학습 모델(Wav2Vec 2.0 등) 실험 불가
  - 하이퍼파라미터 튜닝 횟수 제한
- **영향**: 최적 성능 미달성, 실험 반복 어려움

**3. 데이터 편향**
- 특정 상황(학업 스트레스, 가족 갈등)에 편중된 대화 데이터
- 청소년 외 연령층 적용 시 재학습 필요

**4. 음성 데이터 품질**
- 현상: 실제 상담 데이터가 아닌 정제된 낭독/연기 데이터를 사용하여 감정의 깊이가 얕음.
- 문제점: 우울감을 판단하는 핵심 요소인 '운율(Intonation)', '강세(Stress)', '발화 속도 변화' 등의 음향적 특징이 뚜렷하지 않아 모델 학습에 난항.

---

### 향후 개선 방향

**1. 멀티모달 성능 개선** 🎯 (최우선)
- **음성 모델 고도화**:
  - Wav2Vec 2.0, HuBERT 등 음성 사전학습 모델 Fine-tuning

- **균형 잡힌 융합 전략**:
  - 후기 융합 대신 중기 융합 실험
  - 텍스트와 음성의 Cross-modal attention

**2. 컴퓨팅 리소스 확보**
- **GPU 환경 개선**:
  - Google Colab Pro 또는 AWS/GCP GPU 인스턴스 활용
  - 모델 경량화 (LoRA, Adapter 기법)
  - Mixed Precision Training (FP16) 활용

**3. 실시간 모니터링 시스템**
- 장기 대화 추적을 통한 우울 경향 변화 감지
- 개인화된 베이스라인 설정 및 이상 탐지
- 시계열 분석 기법 적용

---

## 📮 문의

프로젝트 관련 문의사항은 아래로 연락 주시기 바랍니다:

| | |
|---|---|
| 📧 **Email** | xogur1578@gmail.com |
| 🐙 **GitHub** | [github.com/TaehyukRyu/depression-detection](https://github.com/TaehyukRyu/depression-detection) |
| 🚀 **Live Demo** | http://43.201.10.85:8501 |

---

