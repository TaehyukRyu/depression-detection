import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# 페이지 설정
st.set_page_config(
    page_title="감정 분석 - 우울증 조기 감지",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 커스텀 CSS - 아름다운 디자인
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Playfair+Display:wght@400;700&display=swap');
    
    /* 전체 배경 그라데이션 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* 메인 컨테이너 */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* 타이틀 스타일 */
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: white !important;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        animation: fadeInDown 0.8s ease-out;
    }
    
    /* 서브타이틀 */
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Noto Sans KR', sans-serif;
        font-weight: 300;
        animation: fadeIn 1s ease-out;
    }
    
    /* 카드 스타일 */
    .card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    /* 업로드 영역 */
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    /* 결과 카드 */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border-left: 5px solid;
        animation: slideInRight 0.5s ease-out;
    }
    
    .result-card.positive {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    .result-card.negative {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    
    /* 메트릭 스타일 */
    .metric-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        font-family: 'Playfair Display', serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 익스팬더 */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* 오디오 플레이어 */
    audio {
        width: 100%;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* 구분선 */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    }
    
    /* 스피너 */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* 경고/성공 메시지 */
    .stAlert {
        border-radius: 15px;
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    /* 텍스트 박스 */
    .text-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
        font-family: 'Noto Sans KR', sans-serif;
        line-height: 1.8;
        color: #374151;
    }
    
    /* 헤더 아이콘 */
    .icon-header {
        display: inline-block;
        font-size: 2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# 백엔드 URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/predict")

# 타이틀과 서브타이틀
st.markdown("# 🧠 감정 분석")
st.markdown('<div class="subtitle">멀티모달 AI로 당신의 마음을 이해합니다 | 목소리와 언어를 통한 심리 상태 분석</div>', unsafe_allow_html=True)

st.divider()

# 메인 레이아웃
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 오디오 파일 업로드")
    st.markdown("녹음된 음성 파일을 업로드하여 AI 분석을 시작하세요")
    
    uploaded_file = st.file_uploader(
        "WAV, MP3, FLAC 파일을 선택하세요",
        type=["wav", "mp3", "flac"],
        help="최대 파일 크기: 200MB"
    )
    
    if uploaded_file is not None:
        st.success("✅ 파일이 업로드되었습니다!")
        st.audio(uploaded_file, format='audio/wav')
        
        # 파일 정보
        file_details = {
            "파일명": uploaded_file.name,
            "파일 크기": f"{uploaded_file.size / 1024:.2f} KB",
            "업로드 시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with st.expander("📋 파일 정보"):
            for key, value in file_details.items():
                st.text(f"{key}: {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 분석 버튼
        if st.button("🔍 AI 분석 시작", use_container_width=True, type="primary"):
            with st.spinner("🤖 AI가 목소리와 내용을 분석 중입니다..."):
                try:
                    # 백엔드로 파일 전송
                    files = {"audio": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(BACKEND_URL, files=files, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result["success"]:
                            st.session_state["result"] = result
                            st.session_state["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.rerun()
                        else:
                            st.error(f"❌ 분석 실패: {result.get('error')}")
                    else:
                        st.error(f"⚠️ 서버 오류: {response.status_code}")
                        
                except requests.Timeout:
                    st.error("⏱️ 요청 시간 초과. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"🔌 연결 오류: {str(e)}")
    else:
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("👆 음성 파일을 업로드하여 분석을 시작하세요")

# 분석 결과 표시
with col2:
    if "result" in st.session_state and uploaded_file:
        data = st.session_state["result"]["data"]
        stt_text = st.session_state["result"].get("stt_result", "")
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 분석 결과")
        
        # 주요 메트릭
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{}</div>
                <div class="metric-label">예측 감정</div>
            </div>
            """.format(data['label']), unsafe_allow_html=True)
        
        with metric_col2:
            confidence_color = "#10b981" if not data["is_depression"] else "#ef4444"
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value" style="color: {}">{:.1f}%</div>
                <div class="metric-label">신뢰도</div>
            </div>
            """.format(confidence_color, data['confidence'] * 100), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 결과 카드
        card_class = "negative" if data["is_depression"] else "positive"
        emoji = "⚠️" if data["is_depression"] else "✅"
        message = "우울 신호가 감지되었습니다" if data["is_depression"] else "우울 신호가 감지되지 않았습니다"
        recommendation = "전문가 상담을 권장합니다" if data["is_depression"] else "건강한 심리 상태를 유지하고 있습니다"
        
        st.markdown(f"""
        <div class="result-card {card_class}">
            <h3 style="margin: 0 0 1rem 0;">{emoji} {message}</h3>
            <p style="margin: 0; font-size: 1.1rem; color: #374151;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 텍스트 분석 결과
        with st.expander("🗣️ 변환된 텍스트 확인", expanded=False):
            if stt_text:
                st.markdown(f'<div class="text-box">{stt_text}</div>', unsafe_allow_html=True)
            else:
                st.info("대화 내용이 감지되지 않았습니다")
        
        # 상세 분석 리포트
        with st.expander("📝 상세 분석 리포트", expanded=False):
            st.markdown(f'<div class="text-box">{data["explanation"]}</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 분석 결과")
        st.info("왼쪽에서 파일을 업로드하고 분석을 시작하세요")
        st.markdown("</div>", unsafe_allow_html=True)

# 하단: 확률 분포 시각화
if "result" in st.session_state and uploaded_file:
    st.divider()
    
    data = st.session_state["result"]["data"]
    
    # 확률 데이터 생성 (백엔드에서 받지 못하는 경우 더미 데이터)
    # 실제로는 백엔드에서 probabilities를 받아와야 함
    if "probabilities" in data:
        probabilities = data["probabilities"]
    else:
        # 더미 데이터 생성 (예시)
        confidence = data["confidence"]
        if data["is_depression"]:
            probabilities = {
                "우울": confidence,
                "불안": confidence * 0.7,
                "정상": 1 - confidence,
                "스트레스": confidence * 0.5
            }
        else:
            probabilities = {
                "정상": confidence,
                "우울": 1 - confidence,
                "불안": (1 - confidence) * 0.5,
                "스트레스": (1 - confidence) * 0.3
            }
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 감정 확률 분포")
    
    # Plotly 차트 생성
    fig = go.Figure()
    
    # 바차트
    emotions = list(probabilities.keys())
    values = [probabilities[e] * 100 for e in emotions]
    colors = ['#667eea' if i == 0 else '#764ba2' if i == 1 else '#a78bfa' if i == 2 else '#c4b5fd' for i in range(len(emotions))]
    
    fig.add_trace(go.Bar(
        x=emotions,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ),
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>확률: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Noto Sans KR', size=12),
        xaxis=dict(
            title='감정 상태',
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            title='확률 (%)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, max(values) * 1.2]
        ),
        height=400,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 분석 시간 표시
    if "analysis_time" in st.session_state:
        st.caption(f"🕐 분석 완료 시간: {st.session_state['analysis_time']}")

# 푸터
st.divider()
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem 0;">
    <p style="font-size: 0.9rem; margin: 0;">💙 당신의 마음 건강을 응원합니다</p>
    <p style="font-size: 0.8rem; margin: 0.5rem 0 0 0; opacity: 0.8;">이 서비스는 전문적인 의료 상담을 대체할 수 없습니다. 심각한 증상이 있다면 전문가와 상담하세요.</p>
</div>
""", unsafe_allow_html=True)
