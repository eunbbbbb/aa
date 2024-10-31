import streamlit as st
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import random
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(layout="wide")
st.header("🤖 당뇨 운동 추천 챗봇 'EASY당'")
with st.expander(":page_facing_up:  **챗봇 사용 설명서**",expanded=False):
        st.write("""  
                    -  당뇨 환자를 위한 운동 메뉴얼에 대해 질문 할시 크롤링 데이터에서 RAG시스템이 찾아와 답변을 합니다.  출처:http://www.metrohosp.com/sub05/healthguide_contents.php?idx=3020&sub=1\n
                    -  당뇨 확률이 50% 이상인 사용자가 '운동 종류 추천'이라는 글자가 들어가게 질문 할 시 각 카테고리별로 운동 종류와 동영상 링크를 제공합니다. (당뇨 확률이 50% 이하인 사용자는 일반 LLM이 추천) 출처:https://nfa.kspo.or.kr/classroom/program/selectPrescriptionMovieList.kspo\n
                    -  그 외의 질문을 할 시 기본 LLM이 답변을 합니다. 
                 """)


# 세션에 저장된 값 불러오기
data = st.session_state.data_with_percent
# 피처 순서 변경
feature_order = ['age', 'sex','HE_chol', 'HE_wc', 'HE_crea', 'HE_alt', 'HE_TG', 'HE_Upro', 'percent']

# 피처 순서 정렬
data = data[feature_order]
with st.expander(":pill: 사용자의 당뇨 예측 결과",expanded=False):
        st.write(data)


# 임베딩 모델 초기화
def initialize_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )
    except Exception as e:
        st.error(f"임베딩 모델 초기화 중 오류 발생: {e}")
        return None

embeddings_model = initialize_embedding_model()


with open('C:/Users/SAMSUNG/[은비]/[공공데이터 분석및 AI챗봇 과정]/팀프로젝트 3차 대시보드/대시보드/pages/__func__/crawled_data.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 텍스트 분할 및 임베딩
def split_and_embed_text(content, embeddings_model):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(
        documents,
        embedding=embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )

vectorstore = split_and_embed_text(content, embeddings_model)

# LLM 모델 초기화
llm = Ollama(model="gemma2")

# 프롬프트 템플릿 설정
prompt_template = """
### [INST]
지침: 당신의 지식을 바탕으로 질문에 한글로 답하십시오. 
도움이 될 수 있는 맥락은 다음과 같습니다:

{context}

### QUESTION:
{question}

[/INST]
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# 운동 데이터 로드
file_path = 'C:/Users/SAMSUNG/[은비]/[공공데이터 분석및 AI챗봇 과정]/팀프로젝트 3차 대시보드/대시보드/pages/__func__/exercise.csv'
exercise_data = pd.read_csv(file_path)

# 나이에 따른 목표 그룹 매핑
def map_target(user_age):
    if 11 <= user_age <= 12:
        return '유소년'
    elif 13 <= user_age <= 18:
        return '청소년'
    elif 19 <= user_age <= 64:
        return '성인'
    elif user_age >= 65:
        return '어르신'
    else:
        return '알 수 없음'

# 운동 추천 함수
def recommend_exercise(user_age):
    # 사용자 나이에 따른 목표 그룹 설정 (이 함수는 이미 정의되어 있다고 가정)
    target_group = map_target(user_age)

    # 추천할 운동 검색
    filtered_exercises = []
    for _, row in exercise_data.iterrows():
        # row에서 필요한 값 가져오기
        exercise_name = row['Exercise_name']
        fitness_category = row['Fitness_category']
        target = row['Target']
        video_url = row['Video_URL']

        # 사용자 목표 그룹과 일치하는 운동 추가
        if target == target_group:
            filtered_exercises.append({
                'exercise_name': exercise_name,
                'fitness_category': fitness_category,
                'video_url': video_url
            })

    # 카테고리별 랜덤 추천
    category_dict = {}
    for exercise in filtered_exercises:
        category = exercise['fitness_category']
        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append(exercise)

    # 카테고리별 랜덤으로 운동 선택
    recommendations = []
    for category, exercises in category_dict.items():
        if exercises:  # 해당 카테고리에 운동이 있는 경우
            recommendations.append(random.choice(exercises))

    # 추천 결과 반환
    if recommendations:
        result = f"{target_group}을 위한 추천 운동:\n"
        for rec in recommendations:
            result += f"운동명: {rec['exercise_name']}, 카테고리: {rec['fitness_category']}, 비디오 링크: {rec['video_url']}\n"
        return result
    else:
        return "추천할 운동이 없습니다."


# vectorstore를 사용하도록 변경
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff',
                                 retriever=vectorstore.as_retriever(
                                     search_type='mmr',
                                     search_kwargs={'k': 3, 'fetch_k': 10}),
                                 return_source_documents=True)


# 쿼리 처리 함수 

def process_query(user_query, user_age, percent):
    # "운동 종류 추천"이 포함된 경우
    if "운동 종류 추천" in user_query:
        if percent >= 50:
            exercise_recommendations = recommend_exercise(user_age)
            response = prompt | llm
            result = response.invoke({
                "context": """당뇨인에게 적합한 운동을 추천해주는 시스템입니다.
                    이 시스템은 당뇨인의 나이에 따라 운동종류, 카테고리, 비디오 링크를 필수적으로 제공합니다.
                    당뇨인의 운동시 주의사항을 간략하게 제공합니다.""",
                "question": exercise_recommendations
            })
            return f"*당신의 당뇨 위험도는 ({percent:.2f}%)입니다. 추천 운동을 제공합니다.* \n{result}"
        else:
            response = prompt | llm
            result = response.invoke({
                "context": "한글로 답변하세요. 당뇨 위험이 낮은 사용자입니다.",
                "question": user_query
            })
            return f"*당신의 당뇨 위험은 낮습니다 ({percent:.2f}%). 일반적인 건강 조언을 제공합니다.* \n{result}"
    # 벡터스토어에 저장된 데이터에서 찾아서 답변
    result = qa({
        "query": user_query,
        "context": "한글로 답변하세요."})
    negative_keywords = [
    "cannot give", "can't give", "no information", "not available", "cannot answer",
    "unable to provide", "not possible", "insufficient data", "lack of information","can't recommend","cannot recommend"
]

    if result['result']:
      if any(keyword in result['result'].lower() for keyword in negative_keywords):
          # LLM을 호출하여 새로운 결과를 가져옴
          response = prompt | llm
          result = response.invoke({
              "context": "한글로 성의껏 답변하세요.",
              "question": user_query  # query를 question으로 전달
          })
          return f"LLM이 질문에 대해 답변합니다. \n{result}"
      return f"RAG 시스템이 질문에 대해 답변합니다. {result['result']} \n출처: {result['source_documents']}"



# 세션 상태 초기화
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자 입력 폼
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('질문을 입력해주세요: ', '', key='input')
    submitted = st.form_submit_button('물어보기')


user_age = int(data['age'][0])
percent = float(data['percent'][0])  # percent 값 추출

if submitted and user_input:
    with st.spinner("챗봇이 답변 생각 중..."):
        try:
            output = process_query(user_input, user_age, percent)  # percent 전달
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        except Exception as e:
            st.error(f"쿼리 처리 중 오류 발생: {e}")


# 이전 대화 출력
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.write(f"**사용자**: {st.session_state['past'][i]}")
        st.write(f"**EASY당**: {st.session_state['generated'][i]}")
        st.header(" ")
