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
    
    def extract_mfcc(self, audio_path):

        # 음성 로드
        data, sr = librosa.load(audio_path, sr=self.sr)
        
        # MFCC 추출
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=self.n_mfcc)
        
        # 패딩 또는 자르기
        mfcc_padded = self._pad_mfcc(mfcc)
        
        return mfcc_padded
    
    def _pad_mfcc(self, mfcc_array):
        current_length = mfcc_array.shape[1]
        
        if current_length < self.max_time_steps:
        
            pad_width = ((0, 0), (0, self.max_time_steps - current_length))
            return np.pad(mfcc_array, pad_width, mode='constant', constant_values=0)
        else:
       
            return mfcc_array[:, :self.max_time_steps]


class TextPreprocessor:
    
    def __init__(self, model_name='klue/roberta-base', max_length=512, device='cuda'):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.max_length = max_length
        self.device = device
        
        # PHQ-9 증상 키워드
        self.phq9_keywords = [
            # direct_core
            "자살 생각이 든다", "죽고싶다", "죽는게 낫다",
            "자해 생각이 든다", "자해할 생각이다",
            # core_list
            "우울하다", "희망이 없다", "즐겁지 않다",
            "행동이 느려졌다", "말이 느려졌다",
            "일상에 집중을 못한다", "실패했다",
            "가족을 실망 시켰다", "기운이 없다", "피곤하다",
            # indirect_list
            "흥미가 없다", "흥미가 떨어지다",
            "식욕이 줄다", "입맛이 없다", "많이 먹다",
            "잠들기 어렵다", "잠을 너무 많이 잔다",
            "가만히 있질 못한다", "안절부절하다", "잘못하고있다"
        ]
        
        # PHQ-9 키워드 임베딩 (초기화 시 한 번만)
        self.phq9_embeddings = self._encode_phq9_keywords()
    
    def tokenize(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
    
    def calculate_phq9_similarity(self, text):

        # 텍스트 임베딩
        text_emb = self._encode_text(text)
        
        # 코사인 유사도 계산
        cos_sim = (text_emb @ self.phq9_embeddings.T).cpu().numpy()
        max_sim = cos_sim.max()
        
        return float(max_sim)
    
    def _encode_text(self, text):

        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            output = self.model(input_ids, attention_mask)
            last_hidden_state = output.last_hidden_state
            
            mask = attention_mask.unsqueeze(-1).float()
            mean = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            
            mean = F.normalize(mean, p=2, dim=1)
            
            return mean
    
    def _encode_phq9_keywords(self):
        with torch.no_grad():
            embeddings = []
            
            for keyword in self.phq9_keywords:
                emb = self._encode_text(keyword)
                embeddings.append(emb)
            
            embeddings = torch.cat(embeddings, dim=0)
            
            return embeddings


def preprocess_input(text, audio_path, audio_processor, text_processor):

    # 1. 텍스트 처리
    text_encoded = text_processor.tokenize(text)
    phq9_sim = text_processor.calculate_phq9_similarity(text)
    
    # 2. 음성 처리
    mfcc = audio_processor.extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # (1, 30, max_time_steps)
    
    # 3. 텐서 변환
    return {
        'input_ids': text_encoded['input_ids'].unsqueeze(0),  # (1, seq_len)
        'attention_mask': text_encoded['attention_mask'].unsqueeze(0),  # (1, seq_len)
        'phq9_similarity': torch.tensor([[phq9_sim]], dtype=torch.float32),  # (1, 1)
        'mfcc': torch.tensor(mfcc, dtype=torch.float32)  # (1, 1, 30, max_time_steps)
    }


if __name__ == "__main__":
    # 테스트 코드
    print("="*60)
    print("전처리 모듈 테스트")
    print("="*60)
    
    # 1. 음성 전처리 테스트
    print("\n[1] 음성 전처리 테스트")
    audio_proc = AudioPreprocessor()
    print(f"✅ AudioPreprocessor 초기화 완료")
    print(f"   - MFCC 차원: {audio_proc.n_mfcc}")
    print(f"   - 샘플링 레이트: {audio_proc.sr}")
    print(f"   - 최대 시간 길이: {audio_proc.max_time_steps}")
    
    # 2. 텍스트 전처리 테스트
    print("\n[2] 텍스트 전처리 테스트")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_proc = TextPreprocessor(device=device)
    print(f"✅ TextPreprocessor 초기화 완료")
    print(f"   - 모델: klue/roberta-base")
    print(f"   - 최대 길이: {text_proc.max_length}")
    print(f"   - PHQ-9 키워드 수: {len(text_proc.phq9_keywords)}개")
    
    # 3. 샘플 텍스트 테스트
    print("\n[3] 샘플 텍스트 처리")
    sample_text = "요즘 너무 우울하고 아무것도 하기 싫어요"
    similarity = text_proc.calculate_phq9_similarity(sample_text)
    print(f"   입력: {sample_text}")
    print(f"   PHQ-9 유사도: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("전처리 모듈 테스트 완료!")
    print("="*60)
