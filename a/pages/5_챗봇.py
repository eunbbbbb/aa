from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import pandas as pd
import subprocess
import re
import numpy as np

# 허깅페이스의 모델로 임베딩모델 지정
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli', # 사용할 모델 지정(한국어 자연어 추론에 최적화된 모델)
    model_kwargs={'device':'cpu'},  # 모델이 CPU에서 실행되도록 설정(GPU를 사용하려면 'cuda')
    encode_kwargs={'normalize_embeddings':True},  # 임베딩을 정규화하여 모든 벡터가 같은 범위의 값을 갖도록 일관성을 높여줌
)

# 텍스트 데이터 줄글로 가져오기
with open('a/pages/__func__/crawled_data.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 텍스트를 청크로 나누기
# CharacterTextSplitter-> 줄바꿈을 기준(한개)으로 문단 나눔(맥스 토큰이 넘을 수 있다)
# recursiveCharacterTextSplitter-> 줄바꿈,마침표, 쉼표등 기준(여러개)으로 문단 나눔(맥스 토큰을 거의 넘지 않음)
# 현재 데이터 같은 경우 CharacterTextSplitter사용시 27정도 (30개 안 넘음), recursive는 79로(chunk_size=300) 많이 바뀜
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# Document 객체로 변환
documents = [Document(page_content=chunk) for chunk in chunks]

# 각 Document 객체에서 텍스트 추출 후 임베딩 생성
embeds = [embeddings_model.embed_query(doc.page_content) for doc in documents]

# from_documents: 주어진 문서들로부터 벡터 스토어를 생성하는 클래스 메서드
vectorstore = FAISS.from_documents(documents,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE # 벡터 간의 유사성을 측정하는 방법을 설정(코사)
                                  )

# 각 문자열을 Document 객체로 변환
docs = [Document(page_content=chunk) for chunk in chunks]

# FAISS 인스턴스 생성 (임베딩 후 faiss에 저장)
db = FAISS.from_documents(docs, embeddings_model)

# 벡터디비에 저장한 거 로컬에 저장
db.save_local('faiss')



# Git 명령어 실행
subprocess.run(["git", "clone", "https://github.com/ollama/ollama.git"])

# llm과 같이 실행 
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="gemma2")

qa = RetrievalQA.from_chain_type(llm = llm,
                                 chain_type = 'stuff',
                                 retriever = db.as_retriever(
                                     search_type = 'mmr',
                                     search_kwargs = {'k':3, 'fetch_k':10}),
                                 return_source_documents = True)


# 나머지 코드
query = '운동 시 주의사항에 대해 알려줘.'
formatted_query = f"{query} (단, 답변은 반드시 한국어로 작성해 주세요.)"

try:
    result = qa({"query": formatted_query})
    print(result)
except Exception as e:
    print(f"Error occurred: {e}")
