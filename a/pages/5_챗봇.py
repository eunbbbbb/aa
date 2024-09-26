import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import subprocess

# Streamlit 애플리케이션 제목
st.title("챗봇과 대화하기")

# 허깅페이스의 모델로 임베딩모델 지정
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# 텍스트 데이터 줄글로 가져오기
with open('a/pages/__func__/crawled_data.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 텍스트를 청크로 나누기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# Document 객체로 변환
documents = [Document(page_content=chunk) for chunk in chunks]

# FAISS 인스턴스 생성
db = FAISS.from_documents(documents, embeddings_model)
db.save_local('faiss')  # 로컬에 저장

# Git 명령어 실행 (필요한 경우)
# subprocess.run(["git", "clone", "https://github.com/ollama/ollama.git"])

# LLM과 같이 실행
llm = Ollama(model="gemma2")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 10}
    ),
    return_source_documents=True
)

# 사용자 입력 처리
user_input = st.text_input("당신의 질문을 입력하세요:")

if st.button("질문하기"):
    if user_input:
        formatted_query = f"{user_input} (단, 답변은 반드시 한국어로 작성해 주세요.)"
        try:
            result = qa({"query": formatted_query})
            st.write("챗봇의 응답:", result["generated_text"])
        except Exception as e:
            st.error(f"오류 발생: {e}")
    else:
        st.warning("질문을 입력해 주세요.")
