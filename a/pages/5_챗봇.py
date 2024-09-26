import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import requests

# Streamlit 애플리케이션 제목
st.title("🤖 챗봇과 대화하기")

# 사용자 입력 처리
user_input = st.text_input("당신의 질문을 입력하세요:")

if st.button("질문하기"):
    if user_input:
        # Flask API 호출
        url = "http://<ngrok-url>.ngrok.io/api/generate"  # ngrok URL을 여기에 입력하세요
        payload = {"query": user_input}
        
        try:
            response = requests.post(url, json=payload)
            response_data = response.json()
            answer = response_data.get("response", "응답을 받지 못했습니다.")

            # 응답 출력
            st.write(f"**챗봇:** {answer}")
        
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
    else:
        st.warning("질문을 입력해 주세요.")
