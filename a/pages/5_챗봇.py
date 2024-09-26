import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings  # 변경된 import
from langchain_community.vectorstores import FAISS  # 변경된 import
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Streamlit 애플리케이션 제목
st.title("🤖 챗봇과 대화하기")

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

# 질문과 답변이 다 보일 수 있게 저장하는 리스트 
if 'generated' not in st.session_state:  # 챗봇 응답 저장
    st.session_state['generated'] = []
if 'past' not in st.session_state:  # 사용자 입력 저장
    st.session_state['past'] = []

# 사용자 입력 처리
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신의 질문을 입력하세요:', key='input')
    submitted = st.form_submit_button('질문하기')

if submitted and user_input:
    formatted_query = f"{user_input} (단, 답변은 반드시 한국어로 작성해 주세요.)"
    try:
        result = qa({"query": formatted_query})
        st.session_state.past.append(user_input)
        st.session_state.generated.append(result["generated_text"])
    except Exception as e:
        st.error(f"오류 발생: {e}")

# 이전 질문과 답변 표시
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        st.write(f"**You:** {st.session_state['past'][i]}")
        st.write(f"**챗봇:** {st.session_state['generated'][i]}")
