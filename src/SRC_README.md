# 🧠 우울증 조기 감지 시스템 - Python 모듈

## 📁 파일 구조

```
src/
├── preprocessing.py      # 데이터 전처리 모듈
├── model.py             # 모델 구조 정의
├── inference.py         # 추론 메인 모듈
├── usage_examples.py    # 사용 예시 및 API 템플릿
└── requirements.txt     # 의존성 패키지
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 준비

학습된 모델 파일(`.pt`)을 다음 위치에 배치:
```
models/
└── phase3_six_label_all_text_phq9_multimodal.pt
```

### 3. 사용 예시

#### Python 코드에서 사용

```python
from inference import DepressionDetector

# Detector 초기화
detector = DepressionDetector(
    model_path='models/phase3_six_label_all_text_phq9_multimodal.pt',
    device='cuda'  # 또는 'cpu'
)

# 예측 수행
result = detector.predict(
    text="요즘 너무 우울하고 아무것도 하기 싫어요",
    audio_path="data/sample.mp3"
)

# 결과 출력
print(detector.explain_prediction(result))
```

#### 명령줄에서 사용

```bash
python inference.py \
    --model models/multimodal.pt \
    --text "요즘 너무 힘들어요" \
    --audio data/sample.mp3 \
    --device cuda
```

---

## 📦 모듈 설명

### preprocessing.py

데이터 전처리 담당:
- `AudioPreprocessor`: 음성 → MFCC 특징 추출
- `TextPreprocessor`: 텍스트 토큰화 + PHQ-9 유사도 계산
- `preprocess_input()`: 통합 전처리 함수

**주요 기능:**
- MP3/WAV 음성 파일 → 30차원 MFCC
- 한글 텍스트 → KLUE RoBERTa 토큰화
- PHQ-9 키워드 유사도 자동 계산

### model.py

모델 구조 정의:
- `LSTMAudioClassifier`: BiLSTM 기반 음성 분류기
- `KLUETextClassifier`: KLUE RoBERTa + PHQ-9 텍스트 분류기
- `MultiModalClassifier`: Late Fusion 멀티모달 결합
- `load_model()`: 저장된 모델 로드 함수

**특징:**
- 부분 파인튜닝 (KLUE 상위 3개 레이어만)
- Adaptive Pooling으로 가변 길이 오디오 처리
- PHQ-9 유사도를 추가 특징으로 활용

### inference.py

추론 메인 모듈:
- `DepressionDetector`: 추론 통합 클래스
- 단일 예측 + 배치 예측 지원
- 결과 해석 및 시각화

**출력 형식:**
```python
{
    'label': '슬픔',                    # 예측 레이블
    'label_id': 5,                      # 레이블 ID
    'probabilities': {...},             # 클래스별 확률
    'is_depression': True,              # 우울 신호 여부
    'depression_prob': 0.924,           # 우울 확률
    'confidence': 0.873                 # 신뢰도
}
```

### usage_examples.py

향후 구현을 위한 참고 코드:
- FastAPI 서버 템플릿
- Streamlit 데모 템플릿
- Docker 배포 예시
- 다양한 사용 시나리오

---

## ⚙️ 설정 및 파라미터

### 전처리 파라미터

```python
# 음성
AudioPreprocessor(
    n_mfcc=30,           # MFCC 차원
    sr=16000,            # 샘플링 레이트 (Hz)
    max_time_steps=8144  # 최대 시간 길이
)

# 텍스트
TextPreprocessor(
    model_name='klue/roberta-base',
    max_length=512,      # 최대 토큰 길이
    device='cuda'
)
```

### 모델 파라미터

```python
MultiModalClassifier(
    num_class=6,                    # 분류 클래스 수
    text_num_layers_to_train=3,     # 파인튜닝 레이어 수
    audio_input_size=30,            # MFCC 차원
    audio_hidden_size=128,          # LSTM 은닉층 크기
    audio_target_length=200         # LSTM 입력 시퀀스 길이
)
```

---

## 🎯 입력 데이터 요구사항

### 텍스트
- **형식**: UTF-8 인코딩된 한글 텍스트
- **길이**: 제한 없음 (자동으로 512 토큰으로 잘림)
- **예시**: "요즘 너무 힘들어요. 아무것도 하기 싫고..."

### 음성
- **형식**: MP3 또는 WAV
- **샘플링 레이트**: 자동 변환 (16kHz로)
- **길이**: 제한 없음 (자동으로 8144 프레임으로 조정)
- **화자**: 단일 화자 권장 (학생 응답만)

---

## 📊 출력 해석

### 레이블 정의
```python
0: 기쁨    # 긍정적 감정
1: 당황    # 중립적 감정
2: 분노    # 부정적 감정
3: 불안    # ⚠️ 우울 신호
4: 상처    # ⚠️ 우울 신호
5: 슬픔    # ⚠️ 우울 신호
```

### 우울 신호 판단 기준
- **우울 클래스**: 불안(3), 상처(4), 슬픔(5)
- **경고 기준**: `depression_prob > 0.7`
- **권장 조치**: 전문가 상담

---

## 🔧 테스트

각 모듈 개별 테스트:

```bash
# 전처리 테스트
python preprocessing.py

# 모델 구조 테스트
python model.py

# 추론 테스트
python inference.py
```

예상 출력:
```
✅ AudioPreprocessor 초기화 완료
✅ TextPreprocessor 초기화 완료
✅ 모델 로드 완료
```

---

## 📝 사용 시나리오

### 시나리오 1: 실시간 상담 분석
```python
detector = DepressionDetector('models/multimodal.pt')

# 상담 중 실시간 분석
result = detector.predict(
    text=live_transcript,
    audio_path=recorded_audio
)

if result['is_depression'] and result['depression_prob'] > 0.7:
    alert_counselor()  # 상담사에게 알림
```

### 시나리오 2: 대량 데이터 분석
```python
texts = [...]  # 100개 대화
audios = [...]  # 100개 음성

results = detector.predict_batch(texts, audios)

# 우울 신호 필터링
depression_cases = [
    r for r in results 
    if r['is_depression']
]
```

### 시나리오 3: 모니터링 시스템
```python
# 매일 자동 분석
for date in date_range:
    daily_data = load_data(date)
    results = analyze(daily_data)
    
    if high_risk_detected(results):
        send_alert()  # 관리자 알림
```

---

## ⚠️ 주의사항

### 1. 모델 한계
- 텍스트 길이 512 토큰 제한
- 단일 화자 음성에 최적화
- 학습 데이터의 편향 가능성

### 2. 윤리적 고려사항
- 진단 도구가 아닌 **보조 도구**
- 전문가 판단을 대체할 수 없음
- 개인정보 보호 필수

### 3. 성능
- GPU 사용 권장 (CPU는 약 5배 느림)
- 배치 처리 시 메모리 주의
- 첫 실행 시 모델 로드 시간 소요

---

## 🚀 향후 개선 방향

### 단기 (README의 개선사항 참조)
- KoEDA 기반 데이터 증강
- Attention Fusion 적용
- Early Stopping 구현

### 중기
- FastAPI 서버 구현
- Streamlit 데모 개발
- Docker 컨테이너화

### 장기
- Longformer 모델 전환
- Speaker Diarization 통합
- AWS/Cloud 배포

---

## 📚 참고 자료

- [프로젝트 메인 README](../README.md)
- [한계점 및 개선방향](../docs/개선사항.md)
- [실험 노트북](../notebooks/)

---

## 📞 문의

이슈 또는 질문이 있으시면 GitHub Issues를 통해 문의해주세요.

---

**마지막 업데이트**: 2024.11.24  
**버전**: 1.0.0  
**라이선스**: 포트폴리오 목적
