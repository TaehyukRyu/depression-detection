"""
추론(Inference) 메인 모듈
- 텍스트 + 음성 입력을 받아 우울 신호 예측
"""

import torch
import numpy as np
from preprocessing import AudioPreprocessor, TextPreprocessor, preprocess_input
from model import MultiModalClassifier, load_model


class DepressionDetector:
    """우울증 감지 추론 클래스"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: 학습된 모델 .pt 파일 경로
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 레이블 정의
        self.label_names = ['기쁨', '당황', '분노', '불안', '상처', '슬픔']
        self.depression_classes = [3, 4, 5]  # 불안, 상처, 슬픔
        
        # 전처리기 초기화
        print("📌 전처리기 초기화 중...")
        self.audio_processor = AudioPreprocessor(
            n_mfcc=30,
            sr=16000,
            max_time_steps=8144
        )
        self.text_processor = TextPreprocessor(
            model_name='klue/roberta-base',
            max_length=512,
            device=self.device
        )
        
        # 모델 로드
        print("📌 모델 로드 중...")
        self.model = load_model(
            checkpoint_path=model_path,
            num_class=len(self.label_names),
            device=self.device
        )
        
        print(f"✅ DepressionDetector 초기화 완료 (device: {self.device})")
    
    def predict(self, text, audio_path):
        """
        텍스트 + 음성 입력에 대한 예측 수행
        
        Args:
            text: 입력 텍스트 (str)
            audio_path: 음성 파일 경로 (str)
            
        Returns:
            dict: {
                'label': 예측 레이블 (str),
                'label_id': 레이블 ID (int),
                'probabilities': 각 클래스별 확률 (dict),
                'is_depression': 우울 신호 여부 (bool),
                'depression_prob': 우울 신호 확률 (float),
                'confidence': 예측 신뢰도 (float)
            }
        """
        # 1. 전처리
        inputs = preprocess_input(
            text=text,
            audio_path=audio_path,
            audio_processor=self.audio_processor,
            text_processor=self.text_processor
        )
        
        # 2. 입력 데이터를 device로 이동
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        phq9_sim = inputs['phq9_similarity'].to(self.device)
        mfcc = inputs['mfcc'].to(self.device)
        
        # 3. 모델 추론
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, phq9_sim, mfcc)
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_id].item()
        
        # 4. 결과 정리
        probs_dict = {
            self.label_names[i]: float(probs[0, i].item())
            for i in range(len(self.label_names))
        }
        
        # 우울 신호 여부 판단
        is_depression = pred_id in self.depression_classes
        
        # 우울 신호 확률 (불안 + 상처 + 슬픔 확률 합)
        depression_prob = sum([
            probs[0, cls_id].item() 
            for cls_id in self.depression_classes
        ])
        
        return {
            'label': self.label_names[pred_id],
            'label_id': pred_id,
            'probabilities': probs_dict,
            'is_depression': is_depression,
            'depression_prob': depression_prob,
            'confidence': pred_prob
        }
    
    def predict_batch(self, text_list, audio_path_list):
        """
        배치 예측 (여러 입력을 한 번에 처리)
        
        Args:
            text_list: 텍스트 리스트
            audio_path_list: 음성 파일 경로 리스트
            
        Returns:
            list of dict: 각 입력에 대한 예측 결과
        """
        results = []
        
        for text, audio_path in zip(text_list, audio_path_list):
            result = self.predict(text, audio_path)
            results.append(result)
        
        return results
    
    def explain_prediction(self, result):
        """
        예측 결과를 사람이 읽기 쉽게 설명
        
        Args:
            result: predict() 함수의 반환값
            
        Returns:
            str: 설명 텍스트
        """
        label = result['label']
        confidence = result['confidence']
        is_depression = result['is_depression']
        depression_prob = result['depression_prob']
        
        explanation = f"""
        ╔═══════════════════════════════════════╗
        ║        예측 결과                      ║
        ╠═══════════════════════════════════════╣
        ║ 예측 감정: {label} ({confidence:.1%})
        ║ 우울 신호: {'⚠️  예 (주의 필요)' if is_depression else '✅ 아니오'}
        ║ 우울 확률: {depression_prob:.1%}
        ╠═══════════════════════════════════════╣
        ║ 클래스별 확률                          ║
        """
        
        for emotion, prob in sorted(result['probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 20)
            explanation += f"║  {emotion:4s}: {bar:20s} {prob:.1%}\n"
        
        explanation += "╚═══════════════════════════════════════╝"
        
        if is_depression and depression_prob > 0.7:
            explanation += f"""
        
        ⚠️  경고: 높은 우울 신호가 감지되었습니다.
        전문가 상담을 권장합니다.
        """
        
        return explanation


def main():
    """사용 예시"""
    import argparse
    
    parser = argparse.ArgumentParser(description='우울증 조기 감지 추론')
    parser.add_argument('--model', type=str, required=True, 
                       help='모델 .pt 파일 경로')
    parser.add_argument('--text', type=str, required=True, 
                       help='입력 텍스트')
    parser.add_argument('--audio', type=str, required=True, 
                       help='음성 파일 경로 (.mp3 또는 .wav)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='사용할 디바이스')
    
    args = parser.parse_args()
    
    # 1. Detector 초기화
    detector = DepressionDetector(
        model_path=args.model,
        device=args.device
    )
    
    # 2. 예측 수행
    print("\n🔍 예측 수행 중...")
    result = detector.predict(
        text=args.text,
        audio_path=args.audio
    )
    
    # 3. 결과 출력
    print(detector.explain_prediction(result))
    
    # 4. JSON 형태로도 출력
    import json
    print("\n📄 JSON 결과:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 테스트 모드
    print("="*60)
    print("추론 모듈 테스트")
    print("="*60)
    print("\n실제 사용 예시:")
    print("""
    python inference.py \\
        --model models/phase3_six_label_all_text_phq9_multimodal.pt \\
        --text "요즘 너무 우울하고 아무것도 하기 싫어요" \\
        --audio data/sample.mp3 \\
        --device cuda
    """)
    print("\n또는 Python 코드에서:")
    print("""
    from inference import DepressionDetector
    
    detector = DepressionDetector('models/multimodal.pt')
    result = detector.predict(
        text="요즘 너무 우울하고 아무것도 하기 싫어요",
        audio_path="data/sample.mp3"
    )
    print(result)
    """)
