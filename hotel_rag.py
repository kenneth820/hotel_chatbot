# hotel_rag.py
# ===========================================
# 1) 라이브러리 임포트
# ===========================================
from dotenv import load_dotenv
# import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# (2)대화 기억
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ===========================================
# 2) 환경변수 로드 및 대화 기록
# ===========================================
load_dotenv()
# (2)대화 기록 : 이전 대화 기록 저장
history = []
# ===========================================
# 3) 숙박업소 데이터 수집(샘플)
#    - 우선 샘플 데이터 사용
#    - 향후 실제 API 데이터로 교체 가능
# ===========================================
raw_texts = [
    """
    숙소명: 제주 오션뷰 호텔
    지역: 제주도 서귀포시
    1박 가격: 85000원
    평점: 9.0
    주요 후기: 바다 전망이 아름답고 객실이 깨끗하다.
    장점: 오션뷰, 청결도 우수, 친절한 직원
    단점: 주변 대중교통 이용이 다소 불편하다.
    """,

    """
    숙소명: 제주 시티 호텔
    지역: 제주도 제주시
    1박 가격: 70000원
    평점: 8.7
    주요 후기: 공항과 가까우며 주변에 식당이 많다.
    장점: 교통 편리, 저렴한 가격, 주변 편의시설
    단점: 객실 크기가 다소 작다.
    """,

    """
    숙소명: 제주 프리미엄 리조트
    지역: 제주도 서귀포시
    1박 가격: 150000원
    평점: 9.5
    주요 후기: 시설이 고급스럽고 직원 서비스가 훌륭하다.
    장점: 고급 시설, 수영장, 훌륭한 서비스
    단점: 숙박 비용이 비싸다.
    """,

    """
    숙소명: 부산 해운대 비즈니스 호텔
    지역: 부산광역시 해운대구
    1박 가격: 90000원
    평점: 8.8
    주요 후기: 해운대 해변과 가까워 관광하기 편리하다.
    장점: 해변 접근성, 주변 관광지, 합리적인 가격
    단점: 성수기에는 가격이 상승한다.
    """,

    """
    숙소명: 부산 광안리 호텔
    지역: 부산광역시 수영구
    1박 가격: 110000원
    평점: 9.2
    주요 후기: 광안대교 전망이 아름답고 청결하다.
    장점: 아름다운 전망, 청결한 객실, 좋은 위치
    단점: 주말에는 주변이 혼잡하다.
    """,

    """
    숙소명: 서울 명동 호텔
    지역: 서울특별시 중구
    1박 가격: 120000원
    평점: 9.1
    주요 후기: 명동 쇼핑거리와 가까우며 교통이 편리하다.
    장점: 쇼핑 편리, 교통 접근성, 친절한 서비스
    단점: 객실이 다소 좁다.
    """
]

# ===========================================
# 4) 텍스트 → Document 변환 및 분할
# ===========================================
documents = [
    Document(
        page_content=text,
        metadata={"source": f"hotel_{i+1}"}
    )
    for i, text in enumerate(raw_texts)
]

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)

split_documents = splitter.split_documents(documents)

# ===========================================
# 5) 임베딩 및 FAISS 벡터 DB 생성
# ===========================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectordb = FAISS.from_documents(
    split_documents,
    embeddings
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

# ===========================================
# 6) LLM 모델 설정
# ===========================================
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.3
)
parser = StrOutputParser()

# ===========================================
# 7) 후속 질문 재작성 프롬프트
# - 후속 질문을 독립적인 검색 질문으로 변환하는 프롬프트
# ===========================================
question_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        너는 숙박 추천 챗봇의 질문 분석 담당자야.

        이전 대화와 현재 질문을 참고해서
        현재 질문을 독립적으로 이해할 수 있는
        검색 질문으로 바꿔줘.

        예시:
        이전 대화: 제주도에서 10만원 이하 숙소를 찾고 있어.
        현재 질문: 그중에서 평점이 가장 높은 곳은?
        검색 질문: 제주도 10만원 이하 숙소 중 평점이 가장 높은 곳은?

        이전 대화와 관련 없는 새로운 질문이면
        현재 질문을 그대로 사용해.

        질문만 출력하고 다른 설명은 하지 마.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

question_rewrite_chain = (
    question_rewrite_prompt
    | llm
    | StrOutputParser()
)

# ===========================================
# 8) 최종 RAG 답변 프롬프트
# ===========================================
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        너는 여행 숙박업소 전문 추천 AI야.

        사용자의 질문에 답할 때 이전 대화와
        제공된 숙박업소 정보 및 후기를 참고해.

        다음 사항을 지켜줘.

        1. 가격, 평점, 위치를 정확하게 비교해.
        2. 사용자의 예산과 조건을 우선적으로 고려해.
        3. 장점과 단점을 함께 설명해.
        4. 검색된 데이터에 없는 정보는 추측하지 마.
        5. 조건에 적합한 숙소를 최대 3곳 추천해.
        6. 적합한 숙소가 없으면 없다고 답변해.
        7. 한국어로 친절하게 답변해.

        [숙박업소 정보]
        {context}
        """
    ),
    # 이전 대화 기록
    MessagesPlaceholder(variable_name="history"),
    # 현재 질문
    ("human", "{question}")
])

# ===========================================
# 9) RAG + 대화 기억 통합 함수
# ===========================================
def rag_answer(question: str, history: list) -> str:

    # (1) 이전 대화를 참고하여 검색 질문 생성
    search_question = question_rewrite_chain.invoke({
        "history": history,
        "question": question
    })

    # (2) 구체화된 질문으로 숙박업소 검색
    docs = retriever.invoke(search_question)

    # (3) 검색 결과를 Context로 변환
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # (4) 이전 대화 + 검색 결과 + 현재 질문으로 답변 생성
    chain = rag_prompt | llm | parser

    answer = chain.invoke({
        "history": history,
        "context": context,
        "question": question
    })

    # (5) 대화 기록 업데이트
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    return answer
