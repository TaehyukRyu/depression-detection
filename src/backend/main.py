import os
import shutil
import uuid
import speech_recognition as sr  # STT 라이브러리 추가
from fastapi import FastAPI, UploadFile, File, HTTPException
from inference import DepressionDetector

app = FastAPI()

# ==========================================
# 1. 모델 초기화
# ==========================================
MODEL_PATH = "models/phase3_six_label_all_text_phq9_multimodal.pt"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

print("🔄 모델을 로드하고 있습니다...")
try:
    detector = DepressionDetector(model_path=MODEL_PATH)
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    detector = None

# ==========================================
# 2. STT 헬퍼 함수 (음성을 텍스트로 변환)
# ==========================================
def audio_to_text(file_path):
    """
    Google Web Speech API를 사용하여 오디오 파일을 텍스트로 변환합니다.
    """
    recognizer = sr.Recognizer()
    
    try:
        # 오디오 파일 읽기
        with sr.AudioFile(file_path) as source:
            # 주변 소음 보정 (선택 사항)
            # recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
        # 구글 API로 인식 (한국어 설정)
        text = recognizer.recognize_google(audio_data, language='ko-KR')
        print(f"🗣️ STT 인식 결과: {text}")
        return text
        
    except sr.UnknownValueError:
        return ""  # 음성을 이해할 수 없음
    except sr.RequestError as e:
        print(f"STT 서비스 오류: {e}")
        return ""  # 구글 API 오류
    except Exception as e:
        print(f"오디오 변환 중 오류: {e}")
        # 포맷 문제일 수 있으므로 빈 문자열 반환
        return ""

# ==========================================
# 3. 엔드포인트 정의
# ==========================================

@app.get("/")
def root():
    return {"message": "Voice-Only Depression Detection Server Running"}

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """
    오직 오디오 파일만 받아서,
    1. STT로 텍스트 추출
    2. 텍스트 + 음성 특징으로 우울증 예측
    """
    
    # 1) 파일 확장자 검사
    # STT 라이브러리는 wav, flac을 가장 잘 지원합니다. (mp3는 추가 설정 필요할 수 있음)
    if not audio.filename.lower().endswith(('.wav', '.flac', '.aiff', '.mp3')):
        raise HTTPException(status_code=400, detail="지원되는 오디오 형식: wav, flac, aiff")

    unique_filename = f"{uuid.uuid4()}_{audio.filename}"
    temp_path = os.path.join(TEMP_DIR, unique_filename)

    try:
        # 2) 파일 저장
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # 3) STT 실행 (음성 -> 텍스트)
        print("📝 음성을 텍스트로 변환 중...")
        stt_text = audio_to_text(temp_path)
        
        if not stt_text:
            # STT 실패 시 처리 (예: 기본값이나 에러 반환)
            # 여기서는 "말씀이 잘 안 들려요"라고 텍스트를 가정하거나 에러를 낼 수 있습니다.
            # 모델 분석을 위해 최소한의 텍스트가 필요하므로 예외 처리
            return {
                "success": False,
                "error": "음성을 텍스트로 변환할 수 없습니다. 더 명확한 목소리로 녹음해주세요."
            }

        # 4) 멀티모달 예측 실행 (추출된 텍스트 + 오디오 경로)
        print(f"🔍 분석 시작 (Text: {stt_text})")
        result = detector.predict(text=stt_text, audio_path=temp_path)
        
        explanation = detector.explain_prediction(result)

        return {
            "success": True,
            "stt_result": stt_text,  # 변환된 텍스트도 같이 알려줌
            "data": {
                "label": result['label'],
                "confidence": result['confidence'],
                "is_depression": result['is_depression'],
                "explanation": explanation
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    finally:
        # 5) 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)