"""
멀티모달 모델 구조 정의
- KLUE RoBERTa 기반 텍스트 모델
- BiLSTM 기반 음성 모델  
- Late Fusion 멀티모달 결합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class LSTMAudioClassifier(nn.Module):
    """음성(MFCC) 분류 모델 - BiLSTM 기반"""
    
    def __init__(self, num_class, input_size=30, hidden_size=128, num_layers=2, 
                 dropout=0.3, target_length=500):
        """
        Args:
            num_class: 분류 클래스 수
            input_size: MFCC 차원 (기본 30)
            hidden_size: LSTM 은닉층 크기
            num_layers: LSTM 레이어 수
            dropout: Dropout 비율
            target_length: Adaptive Pooling 후 시퀀스 길이
        """
        super(LSTMAudioClassifier, self).__init__()
        
        self.target_length = target_length
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_class)  # *2 for bidirectional
    
    def forward(self, mfcc):
        """
        Args:
            mfcc: (batch, 1, 30, time_steps)
            
        Returns:
            y_pred: (batch, num_class) 로짓
        """
        # mfcc shape: (batch, 1, 30, 8144)
        x = mfcc.squeeze(1)  # (batch, 30, 8144)
        
        # Adaptive Average Pooling으로 시퀀스 길이 축소
        x = F.adaptive_avg_pool1d(x, output_size=self.target_length)
        
        # (batch, 30, target_length) → (batch, target_length, 30)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, target_length, hidden*2)
        
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Dropout & Classification
        x = self.dropout(last_output)
        y_pred = self.classifier(x)
        
        return y_pred


class KLUETextClassifier(nn.Module):
    """텍스트 분류 모델 - KLUE RoBERTa + PHQ-9 유사도"""
    
    def __init__(self, num_class, num_layers_to_train=3, dropout=0.3):
        """
        Args:
            num_class: 분류 클래스 수
            num_layers_to_train: 파인튜닝할 상위 레이어 수
            dropout: Dropout 비율
        """
        super(KLUETextClassifier, self).__init__()
        
        # KLUE RoBERTa 로드
        self.klue = AutoModel.from_pretrained('klue/roberta-base')
        config = AutoConfig.from_pretrained('klue/roberta-base')
        hidden_size = config.hidden_size  # 768
        
        # ============================================
        # 부분 파인튜닝 설정
        # ============================================
        total_layers = 12  # KLUE는 12개 레이어
        layers_to_freeze = total_layers - num_layers_to_train
        
        # 1) Embedding layer freeze
        for param in self.klue.embeddings.parameters():
            param.requires_grad = False
        
        # 2) 처음 N개 레이어 freeze
        for i in range(layers_to_freeze):
            for param in self.klue.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # ============================================
        # PHQ-9 유사도 특징 레이어
        self.phq9_feature_layer = nn.Linear(1, 16)
        
        # Dropout & Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + 16, num_class)
    
    def forward(self, input_ids, attention_mask, phq9_features):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            phq9_features: (batch, 1) PHQ-9 유사도 점수
            
        Returns:
            y_pred: (batch, num_class) 로짓
        """
        # KLUE RoBERTa 인코딩
        output = self.klue(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.pooler_output  # (batch, 768)
        
        # PHQ-9 특징 변환
        phq9 = self.phq9_feature_layer(phq9_features)  # (batch, 16)
        phq9 = torch.relu(phq9)
        
        # 특징 결합
        combined = torch.cat([cls_token, phq9], dim=1)  # (batch, 784)
        
        # Dropout & Classification
        x = self.dropout(combined)
        y_pred = self.classifier(x)
        
        return y_pred


class MultiModalClassifier(nn.Module):
    """멀티모달 분류 모델 - Late Fusion"""
    
    def __init__(self, num_class, text_num_layers_to_train=3, 
                 audio_input_size=30, audio_hidden_size=128, audio_target_length=500):
        """
        Args:
            num_class: 분류 클래스 수
            text_num_layers_to_train: 텍스트 모델 파인튜닝 레이어 수
            audio_input_size: MFCC 차원
            audio_hidden_size: LSTM 은닉층 크기
            audio_target_length: LSTM 입력 시퀀스 길이
        """
        super(MultiModalClassifier, self).__init__()
        
        # 텍스트 + PHQ-9 모델
        self.text_model = KLUETextClassifier(
            num_class=num_class,
            num_layers_to_train=text_num_layers_to_train
        )
        
        # 음성(MFCC) 모델
        self.audio_model = LSTMAudioClassifier(
            num_class=num_class,
            input_size=audio_input_size,
            hidden_size=audio_hidden_size,
            target_length=audio_target_length
        )
    
    def forward(self, input_ids, attention_mask, phq9_features, mfcc):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            phq9_features: (batch, 1)
            mfcc: (batch, 1, 30, time_steps)
            
        Returns:
            final_y: (batch, num_class) 최종 로짓
        """
        # 텍스트 모델 예측
        text_y = self.text_model(input_ids, attention_mask, phq9_features)
        
        # 음성 모델 예측
        audio_y = self.audio_model(mfcc)
        
        # Late Fusion (단순 평균)
        final_y = (text_y + audio_y) / 2.0
        
        return final_y


def load_model(checkpoint_path, num_class=6, device='cuda'):
    """
    저장된 모델 가중치 로드
    
    Args:
        checkpoint_path: .pt 파일 경로
        num_class: 분류 클래스 수
        device: 'cuda' or 'cpu'
        
    Returns:
        model: 로드된 모델
    """
    model = MultiModalClassifier(
        num_class=num_class,
        text_num_layers_to_train=3,
        audio_input_size=30,
        audio_hidden_size=128,
        audio_target_length=200
    ).to(device)
    
    # 가중치 로드
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"✅ 모델 로드 완료: {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # 테스트 코드
    print("="*60)
    print("모델 구조 테스트")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_class = 6
    batch_size = 2
    
    # 1. 음성 모델 테스트
    print("\n[1] 음성 모델 (LSTMAudioClassifier)")
    audio_model = LSTMAudioClassifier(num_class=num_class).to(device)
    dummy_mfcc = torch.randn(batch_size, 1, 30, 8144).to(device)
    audio_out = audio_model(dummy_mfcc)
    print(f"   입력 shape: {dummy_mfcc.shape}")
    print(f"   출력 shape: {audio_out.shape}")
    print(f"   ✅ 테스트 통과")
    
    # 2. 텍스트 모델 테스트
    print("\n[2] 텍스트 모델 (KLUETextClassifier)")
    text_model = KLUETextClassifier(num_class=num_class).to(device)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 512)).to(device)
    dummy_attention = torch.ones(batch_size, 512).to(device)
    dummy_phq9 = torch.randn(batch_size, 1).to(device)
    text_out = text_model(dummy_input_ids, dummy_attention, dummy_phq9)
    print(f"   입력 shape: input_ids={dummy_input_ids.shape}, phq9={dummy_phq9.shape}")
    print(f"   출력 shape: {text_out.shape}")
    print(f"   ✅ 테스트 통과")
    
    # 3. 멀티모달 모델 테스트
    print("\n[3] 멀티모달 모델 (MultiModalClassifier)")
    multimodal_model = MultiModalClassifier(num_class=num_class).to(device)
    final_out = multimodal_model(dummy_input_ids, dummy_attention, dummy_phq9, dummy_mfcc)
    print(f"   출력 shape: {final_out.shape}")
    print(f"   ✅ 테스트 통과")
    
    # 4. 파라미터 수 출력
    print("\n[4] 모델 파라미터 수")
    total_params = sum(p.numel() for p in multimodal_model.parameters())
    trainable_params = sum(p.numel() for p in multimodal_model.parameters() if p.requires_grad)
    print(f"   전체 파라미터: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")
    print(f"   Frozen: {total_params - trainable_params:,}")
    
    print("\n" + "="*60)
    print("모델 구조 테스트 완료!")
    print("="*60)
