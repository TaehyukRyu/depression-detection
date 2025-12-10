import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class AudioPreprocessor:
    def __init__(self, n_mfcc=30, sr=16000, max_time_steps=8144):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.max_time_steps = max_time_steps

    def mfcc(self, audio_path):
        # 1. 오디오 로드
        data, _ = librosa.load(audio_path, sr=self.sr)

        # 2. 무음 제거 (Trim)
        data_trim, _ = librosa.effects.trim(data, top_db=20)

        # 3. 볼륨 정규화 (RMS)
        rms = librosa.feature.rms(y=data_trim)
        mean_rms = np.mean(rms)
        epsilon = 1e-10
        target_rms = 0.05
        data_normalized = (data_trim / (mean_rms + epsilon)) * target_rms

        # 4. MFCC 추출
        mfcc = librosa.feature.mfcc(y=data_normalized, sr=self.sr, n_mfcc=self.n_mfcc)
        
        # 5. 패딩 또는 자르기
        mfcc_padded = self.pad_mfcc(mfcc)
        
        # 6. 텐서 변환 (batch, channel, h, w) 형태로 맞추기 위해 차원 추가
        # (30, length) -> (1, 30, length)
        return torch.FloatTensor(mfcc_padded).unsqueeze(0)

    def pad_mfcc(self, mfcc):
        current_length = mfcc.shape[1]
        if current_length < self.max_time_steps:
            # Zero padding
            pad_width = ((0, 0), (0, self.max_time_steps - current_length))
            return np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        else:
            # Truncate
            return mfcc[:, :self.max_time_steps]


class TextPreprocessor:
    def __init__(self, model_name='klue/roberta-base', max_length=512, device='cpu'):
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.phq9_keywords = [
            "자살 생각이 든다", "죽고싶다", "죽는게 낫다", "자해 생각이 든다", 
            "자해할 생각이다", "흥미가 없다", "흥미가 떨어지다", "식욕이 줄다", 
            "입맛이 없다", "많이 먹다", "잠들기 어렵다", "잠을 너무 많이 잔다",
            "가만히 있질 못한다", "안절부절하다", "잘못하고있다", "우울하다", 
            "희망이 없다", "즐겁지 않다", "행동이 느려졌다", "말이 느려졌다",
            "일상에 집중을 못한다", "실패했다", "가족을 실망 시켰다", 
            "기운이 없다", "피곤하다"
        ]
        # PHQ-9 임베딩 미리 계산
        self.phq9_emb = self.encode_texts(self.phq9_keywords)

    def encode_texts(self, texts):
        # 텍스트 리스트를 벡터로 변환
        with torch.no_grad():
            token = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            input_ids = token['input_ids'].to(self.device)
            attention_mask = token['attention_mask'].to(self.device)
            
            output = self.model(input_ids, attention_mask)
            last_hidden_state = output.last_hidden_state
            
            mask = attention_mask.unsqueeze(-1).float()
            mean = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            mean = F.normalize(mean, p=2, dim=1)
            return mean

    def get_inputs(self, text):
        # 1. 기본 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 2. PHQ-9 유사도 계산
        text_emb = self.encode_texts([text]) # (1, 768)
        # 코사인 유사도 (text_emb와 phq9_emb 내적)
        cos_sim = (text_emb @ self.phq9_emb.T) # (1, 25)
        # 가장 높은 유사도 값 추출
        max_sim = cos_sim.max(dim=1)[0].unsqueeze(1) # (1, 1)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'phq9_similarity': max_sim.cpu() # CPU로 이동
        }

def preprocess_input(text, audio_path, audio_processor, text_processor):
    """
    텍스트와 오디오 경로를 받아 모델 입력 형태로 변환하는 통합 함수
    """
    # 1. 텍스트 전처리
    text_inputs = text_processor.get_inputs(text)
    
    # 2. 오디오 전처리
    mfcc_tensor = audio_processor.mfcc(audio_path)
    
    return {
        'input_ids': text_inputs['input_ids'],           # (1, 512)
        'attention_mask': text_inputs['attention_mask'], # (1, 512)
        'phq9_similarity': text_inputs['phq9_similarity'], # (1, 1)
        'mfcc': mfcc_tensor                              # (1, 30, 8144)
    }