# streamlit_app.py
import streamlit as st
from hotel_rag import rag_answer

# ===========================================
# 1) 페이지 기본 설정
# ===========================================
st.set_page_config(
    page_title="AI 숙박 추천 챗봇",
    page_icon="🏨",
    layout="wide"
)

# ===========================================
# 2) 웹사이트 제목
# ===========================================
st.title("🏨 AI 숙박 추천 챗봇")

st.write(
    "RAG 기반 AI가 여행 조건과 숙박 후기를 분석하여 "
    "적합한 숙소를 추천합니다."
)

# ===========================================
# 3) 세션별 대화 기록 초기화
# ===========================================
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===========================================
# 4) 사이드바
# ===========================================
with st.sidebar:
    st.header("검색 안내")
    st.write("현재 제공되는 숙박 지역")
    st.markdown("""
    - 제주도
    - 부산
    - 서울
    """)
    st.divider()
    st.subheader("사용 방법")
    st.write("""
    1. 여행 조건을 질문합니다.
    2. AI가 숙박 정보를 검색합니다.
    3. 추천 결과를 확인합니다.
    4. 추가 질문으로 대화를 이어갑니다.
    """)
    st.divider()

    if st.button("대화 기록 초기화", use_container_width=True):

        st.session_state.history = []
        st.session_state.messages = []

        st.rerun()

# ===========================================
# 5) 기존 대화 내용 화면에 출력
# ===========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===========================================
# 6) 사용자 질문 입력
# ===========================================
question = st.chat_input(
    "예: 제주도에서 10만 원 이하 숙소를 추천해 줘."
)

# ===========================================
# 7) 질문 처리 및 AI 답변 출력
# ===========================================
if question:
    # 사용자 질문 저장
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("숙박 정보를 검색하고 있습니다..."):
            try:
                answer = rag_answer(
                    question,
                    st.session_state.history
                )
                st.markdown(answer)
                # AI 답변 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
