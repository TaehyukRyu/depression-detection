# 청소년 우울 신호 탐지 시스템 (Adolescent Depression Detection System)

<div align="center">

**텍스트와 음성 데이터를 활용한 멀티모달 딥러닝 기반 청소년 우울 신호 조기 탐지 시스템**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30.0-yellow.svg)](https://huggingface.co/transformers/)

</div>

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [시스템 아키텍처](#-시스템-아키텍처)
- [데이터셋](#-데이터셋)
- [실험 결과](#-실험-결과)
- [설치 및 실행](#-설치-및-실행)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🎯 프로젝트 개요

### 배경 및 필요성

기존 감성 분석 모델은 **'단순 슬픔'**과 임상적 주의가 필요한 **'우울'**을 명확히 구분하지 못하는 한계가 있습니다. 특히 청소년의 대화에서는 미묘한 부정 감정과 실제 위험 신호가 혼재되어 나타나므로, 이를 정밀하게 탐지할 수 있는 기술이 필요합니다.

### 프로젝트 목표

- ✅ **우울 신호 특화 탐지**: 단순 부정 감정(짜증, 분노)과 임상적 우울 신호 구분
- ✅ **멀티모달 융합**: 텍스트의 의미 정보와 음성의 비언어적 특징(운율, 톤) 결합
- ✅ **도메인 지식 활용**: PHQ-9(우울증 선별검사) 기반 가중치 적용으로 탐지 민감도 향상
- ✅ **서비스화**: 웹 기반 실시간 진단 인터페이스 구축 및 배포

---

## 🌟 주요 특징

### 1. 커스텀 라벨링 시스템 (PHQ-9 기반)

기존 6개 감정 대분류를 PHQ-9 증상과 연관시켜 4개 커스텀 라벨로 재구성:

| 라벨 | 설명 | 매핑 감정 |
|------|------|-----------|
| **0: 우울 신호** | PHQ-9 증상과 직결되는 감정 | 우울한, 좌절한, 비통한, 염세적인, 고립된, 죄책감의 등 |
| **1: 긍정** | 긍정적 정서 | 기쁨 대분류 전체 |
| **2: 일반 부정** | 일상적 부정 감정 | 짜증, 분노, 질투 등 |
| **3: 중립** | 모호하거나 중립적 감정 | 방어적인, 남의 시선 의식하는 등 |

### 2. PHQ-9 도메인 지식 주입

**의미 유사도 기반 특징 추출**:
```python
# PHQ-9 증상 키워드 벡터와 텍스트 벡터 간 코사인 유사도 계산
similarity_score = cosine_similarity(text_embedding, phq9_embedding)
```

- **Direct Core** (가중치 3.0): 자살/자해 관련 직접적 표현
- **Core List** (가중치 2.0): 우울, 무기력, 집중력 저하 등 핵심 증상
- **Indirect List** (가중치 1.0): 수면/식욕 변화, 불안 등 간접 증상

### 3. 멀티모달 융합 아키텍처

```
📝 텍스트 브랜치              🎤 음성 브랜치
   ↓                            ↓
KLUE-RoBERTa                  MFCC 추출
   ↓                            ↓
[CLS] Token (768dim)          LSTM (128dim)
   ↓                            ↓
PHQ-9 Feature (16dim)              ↓
   ↓                            ↓
   └────── Concatenation ───────┘
                ↓
          Classifier (6 classes)
```

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 인터페이스                       │
│                   (Streamlit Web App)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
              ┌───────▼───────┐
              │   FastAPI     │
              │  REST API     │
              └───────┬───────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
┌─────────┐   ┌─────────────┐   ┌──────────┐
│  Text   │   │   PHQ-9     │   │  Audio   │
│Processor│   │   Feature   │   │ Processor│
└────┬────┘   └──────┬──────┘   └────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
            ┌────────▼────────┐
            │  Multimodal     │
            │   Classifier    │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │  우울 신호 예측   │
            │   + 신뢰도       │
            └─────────────────┘
```

---

## 📊 데이터셋

### 텍스트 데이터
- **출처**: AI Hub 감성 대화 말뭉치 (청소년 필터링)
- **규모**: 10,582건
- **구성**: 발화 텍스트 (3개 턴 대화)
- **전처리**:
  - 비언어적 요소 및 특수문자 제거
  - KLUE-RoBERTa Tokenizer 활용
  - Max Length: 71 tokens (90% percentile)

### 음성 데이터
- **규모**: 2,876건 (멀티모달)
- **형식**: WAV/MP3
- **전처리**:
  ```python
  - Sampling Rate: 16,000Hz
  - MFCC: 30 coefficients
  - 무음 구간 제거 (Trim, top_db=20)
  - RMS 기반 볼륨 정규화 (target_rms=0.05)
  - Max Time Steps: 8,144 frames (90% percentile)
  ```

### 클래스 분포

**6개 대분류** (Phase 1):
```
불안: 1,998건 | 슬픔: 1,887건 | 당황: 1,844건
상처: 1,818건 | 분노: 1,792건 | 기쁨: 1,243건
```

**4개 커스텀 라벨** (Phase 2, 3):
```
일반 부정: 5,775건 | 우울 신호: 3,210건
긍정: 1,243건 | 중립: 354건
```

---

## 📈 실험 결과

### Phase 1: 라벨링 및 문맥 비교 (8개 모델)

**목표**: 최적의 라벨링 시스템 및 입력 전략 선정

| 모델 | 텍스트 | 라벨 | Accuracy | Depression Recall | Depression F1 |
|------|--------|------|----------|-------------------|---------------|
| MLP | All | 6-class | 0.4932 | 0.4845 | 0.6064 |
| **Fine-tuning** | **All** | **6-class** | **0.6907** | **0.8229** | **0.8292** |
| MLP | All | 4-class | 0.6356 | 0.3292 | 0.4125 |
| Fine-tuning | All | 4-class | 0.7288 | 0.7362 | 0.6514 |

**🔍 핵심 발견**:
- ✅ **전체 문장(All)** > 첫 문장(First): 문맥 정보 포함 시 성능 향상
- ✅ **Fine-tuning** > MLP: RoBERTa 전체 파라미터 활용 시 우월
- ✅ **6-class** 최적: 우울 신호(불안+상처+슬픔) 탐지에 가장 효과적

---

### Phase 2: PHQ-9 가중치 효과 (4개 모델)

**목표**: 도메인 지식 주입의 유효성 검증

| 모델 | PHQ-9 | Accuracy | Depression Precision | Depression Recall | Depression F1 |
|------|-------|----------|----------------------|-------------------|---------------|
| 6-class (Baseline) | ❌ | 0.6907 | 0.8356 | 0.8229 | 0.8292 |
| **6-class + PHQ-9** | **✅** | **0.6923** | **0.8317** | **0.8346** | **0.8331** |
| 4-class (Baseline) | ❌ | 0.7288 | 0.5840 | 0.7362 | 0.6514 |
| 4-class + PHQ-9 | ✅ | 0.7276 | 0.5932 | 0.7040 | 0.6439 |

**🔍 핵심 발견**:
- ✅ **Recall 향상**: False Negative 감소 (303건 → 283건 in 6-class)
- ✅ **Precision 유지**: 과탐지 없이 민감도 개선
- ⚠️ **4-class에서는 미미**: 이미 충분한 특징 학습으로 추가 효과 제한적

---

### Phase 3: 멀티모달 융합 (2개 모델)

**목표**: 텍스트 + 음성 시너지 효과 검증

| 모델 | Modality | Accuracy | Depression Precision | Depression Recall | Depression F1 |
|------|----------|----------|----------------------|-------------------|---------------|
| Text Only | 📝 | 0.7115 | 0.9459 | 0.9016 | 0.9232 |
| **Multimodal** | **📝+🎤** | **0.6466** | **0.9742** | **0.8266** | **0.8943** |

**🔍 핵심 발견**:
- ⚠️ **Accuracy 하락**: 음성 데이터 노이즈 및 품질 편차 영향
- ✅ **Precision 향상**: 0.9459 → **0.9742** (False Positive 대폭 감소)
- ⚠️ **Recall 하락**: 텍스트 모델이 더 민감한 탐지 수행

**💡 운영 전략**:
- **메인 모델**: Text + PHQ-9 (안정성 ↑)
- **보조 모델**: Multimodal (정밀도 ↑, 2차 검증용)

---

### 최종 성능 요약

```
✨ 최종 선정 모델: Text (All) + PHQ-9 + 6-class Fine-tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Overall Accuracy:          69.23%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 우울 신호 탐지 성능 (불안+상처+슬픔):
   - Precision:  83.17%  (False Positive 최소화)
   - Recall:     83.46%  (실제 위험군 놓치지 않음)
   - F1-Score:   83.31%  (균형잡힌 성능)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ False Negative: 283/1,711건 (16.5%)
✅ True Positive:  1,428/1,711건 (83.5%)
```

---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/yourusername/adolescent-depression-detection.git
cd adolescent-depression-detection

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# AI Hub에서 감성 대화 말뭉치 다운로드 후
mkdir data
# data/ 폴더에 배치
```

### 3. 모델 학습

```bash
# Phase 1: 라벨링 비교
python train_phase1.py --label_type 6class --text_type all

# Phase 2: PHQ-9 효과 검증
python train_phase2.py --use_phq9 True

# Phase 3: 멀티모달
python train_phase3.py --use_audio True
```

### 4. 추론

```python
from model import DepressionDetector

detector = DepressionDetector()
text = "요즘 아무것도 하기 싫고 자꾸 죽고 싶다는 생각이 들어"
result = detector.predict(text)

print(f"우울 신호 확률: {result['depression_prob']:.2%}")
print(f"권장 조치: {result['recommendation']}")
```

---

## 🛠 기술 스택

### AI/ML

| 기술 | 용도 | 버전 |
|------|------|------|
| **PyTorch** | 딥러닝 프레임워크 | 2.0.1 |
| **Transformers** | KLUE-RoBERTa 모델 | 4.30.0 |
| **Librosa** | 음성 특징 추출 (MFCC) | 0.10.0 |
| **scikit-learn** | 데이터 전처리 및 평가 | 1.3.0 |
| **sentence-transformers** | PHQ-9 임베딩 | 2.2.2 |

### Backend/Ops

| 기술 | 용도 |
|------|------|
| **FastAPI** | REST API 서버 |
| **Docker** | 컨테이너화 |
| **AWS EC2/S3** | 클라우드 배포 |

### Frontend

| 기술 | 용도 |
|------|------|
| **Streamlit** | 웹 인터페이스 |

---

## 📁 프로젝트 구조

```
adolescent-depression-detection/
│
├── data/                          # 데이터셋
│   ├── Training_221115_add/       # AI Hub 원본 데이터
│   ├── six_label_all_text.csv     # Phase 1 전처리 결과
│   ├── four_label_all_text_phq9.csv  # Phase 2 데이터
│   └── 음성데이터전처리_*.pkl     # Phase 3 MFCC 데이터
│
├── models/                        # 학습된 모델
│   ├── phase1_six_label_all_text_finetuning.pt
│   ├── phase2_six_label_all_text_phq9.pt
│   └── phase3_multimodal.pt
│
├── notebooks/                     # 실험 노트북
│   ├── Phase_1_라벨_문맥_모델_비교.ipynb
│   ├── Phase_2_PHQ9_가중치_효과.ipynb
│   └── Phase_3_멀티모달.ipynb
│
├── src/                           # 소스 코드
│   ├── preprocessing/
│   │   ├── text_preprocessing.py
│   │   └── audio_preprocessing.py
│   ├── models/
│   │   ├── text_model.py
│   │   ├── audio_model.py
│   │   └── multimodal_model.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       └── metrics.py
│
├── api/                           # FastAPI 서버
│   ├── main.py
│   └── schemas.py
│
├── app/                           # Streamlit 앱
│   └── streamlit_app.py
│
├── tests/                         # 테스트 코드
│
├── requirements.txt               # 의존성 패키지
├── Dockerfile                     # Docker 설정
├── README.md                      # 본 문서
└── 프로젝트_계획서.docx             # 상세 계획서
```

---

## 📊 실험 상세 결과

### Confusion Matrix (Phase 2 - 최종 모델)

```
실제 ↓ / 예측 →    기쁨   당황   분노   불안   상처   슬픔
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기쁨                343    6     10    1     5     8
당황                3      394   30    39    44    43
분노                3      39    347   43    50    56
불안                4      57    52    370   66    51  ← 우울 신호
상처                9      37    52    28    351   68  ← 우울 신호
슬픔                4      32    36    32    69    393 ← 우울 신호
```

**핵심 관찰**:
- ✅ 우울 신호 클래스 간 혼동은 **허용 가능** (모두 위험군)
- ✅ 우울 신호 → 긍정/중립 오분류 **최소화** (20건 이하)
- ⚠️ 분노 ↔ 슬픔 혼동 존재 (감정 표현 방식 차이)

---

## 🎓 연구 의의

### 기술적 기여

1. **도메인 지식 통합 방법론**: PHQ-9 증상을 임베딩 유사도로 정량화하여 모델에 주입
2. **계층적 라벨링 전략**: 임상 기준에 따른 커스텀 분류 체계 구축
3. **멀티모달 융합 최적화**: 텍스트 중심 + 음성 보조 구조로 안정성 확보

### 사회적 기여

- **조기 개입 지원**: 상담사가 놓칠 수 있는 미묘한 우울 징후 포착
- **객관적 평가 도구**: 주관적 판단 보완을 위한 정량적 지표 제공
- **접근성 향상**: 비대면 상담 플랫폼에 통합 가능한 자동화 시스템

---

## 📌 제한사항 및 향후 과제

### 현재 제한사항

- ⚠️ **데이터 편향**: 특정 상황(학업, 가족)에 편중된 대화 데이터
- ⚠️ **음성 품질**: 녹음 환경 차이로 인한 MFCC 변동성
- ⚠️ **일반화**: 청소년 외 연령층 적용 시 재학습 필요

### 향후 개선 방향

1. **데이터 증강**:
   - 다양한 상황 시나리오 추가 수집
   - Back-translation, Paraphrasing 기법 적용

2. **모델 고도화**:
   - Attention 메커니즘으로 중요 구간 식별
   - Wav2Vec 2.0 등 최신 음성 모델 도입

3. **실시간 모니터링**:
   - 장기 대화 추적을 통한 우울 경향 변화 감지
   - 개인화된 베이스라인 설정 및 이상 탐지

4. **윤리적 고려**:
   - 프라이버시 보호 강화 (On-device 추론)
   - 편향성 검증 및 공정성 평가

---

## 📄 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

## 👥 팀 소개

- **개발자**: [이름]
- **지도교수**: [이름]
- **소속**: [대학/기관]

---

## 📮 문의

프로젝트 관련 문의사항은 아래로 연락 주시기 바랍니다:

- 📧 Email: your.email@example.com
- 🐙 GitHub Issues: [링크]

---

## 🙏 감사의 말

- **AI Hub**: 감성 대화 말뭉치 데이터 제공
- **KLUE Team**: KLUE-RoBERTa 사전학습 모델
- **Hugging Face**: Transformers 라이브러리

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요! ⭐**

</div>
