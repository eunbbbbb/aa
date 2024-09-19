import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pages.__func__.function import remove_outliers

# 페이지 설정
st.set_page_config(layout="wide")

# 데이터 로드
data = joblib.load(r'C:\Users\1104_6\OneDrive\바탕 화면\고은비 팀프로젝트 관련 논문\10년치데이터.pkl')
# 시스템에서 사용할 수 있는 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'

## main ##
with st.container():
    tab1, tab2, tab3= st.tabs(['사용 데이터', "당뇨 진단 기준", "당뇨 현황"])

with tab1: # 국건영 소개
    con1, con2= st.columns([0.4, 0.6])

    st.subheader('이 조사는 보통 보건복지부와 질병관리청, 그리고 관련 연구 기관들이 협력하여 진행합니다.')

    with con1:
        st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/국건영.png', caption='출처: 질병관리청')
    with con2:
        st.header(':pill: 국민 건강 영양 조사란?')
        with st.expander(" ",expanded=True):
            st.write('''
                    ##### :small_blue_diamond: 한국인의 건강 상태와 식습관을 분석하여, 국민 건강 증진을 위한 정책 및 프로그램을 개발하는 데 필요한 기초 자료를 제공
                    
                    :small_blue_diamond: 조사 내용
                    - 건강 상태: 개인의 건강 이력, 질병 유병률, 신체 지표(체중, 신장, 혈압 등) 등을 조사
                    - 영양 섭취: 식단 분석을 통해 영양 불균형 또는 과잉 섭취 문제를 파악
                    - 생활 습관: 운동, 흡연, 음주 등 생활 습관과 관련된 정보를 수집
                    - 사회경제적 정보: 교육 수준, 직업, 소득 등 사회경제적 배경도 조사하여 건강과 영양에 미치는 영향을 분석
                    
                    :small_blue_diamond: 조사 방법:   -설문조사 -신체 측정 -혈액 검사
                        ''')
   
 


with tab2:
    con3, con4 = st.columns([0.4, 0.3])

    with con3:
        st.header(":pill: 당뇨 진단 기준")
        df = data[['HE_glu','HE_HbA1c']]
        df_transformed = remove_outliers(df)

        dang = st.selectbox("당뇨 진단 기준에 따른 데이터 시각화", ["공복혈당", "당화혈색소"])
        if dang == '공복혈당':
            sns.histplot(x='HE_glu', data=df_transformed, kde=False, stat='density', bins=50)
            # 수직선 추가
            plt.axvline(100, label='당뇨 전단계', linestyle='-', color='red', alpha=0.5)
            plt.axvline(126, label='당뇨', linestyle='--', color='red', alpha=1)
            # 범례 추가
            plt.legend()
            plt.title('공복 혈당')
            # 그래프 출력
            st.pyplot(plt)

        elif dang == '당화혈색소':
            sns.histplot(x='HE_HbA1c', data=df_transformed, kde=False, stat='density', bins=48)
            # 수직선 추가
            plt.axvline(5.7, label='당뇨 전단계', linestyle='-', color='red', alpha=0.5)
            plt.axvline(6.5, label='당뇨', linestyle='--', color='red', alpha=1)
            # 범례 추가
            plt.legend()
            plt.title('당화혈색소')
            # 그래프 출력
            st.pyplot(plt)
        
        plt.clf()

        st.markdown('___')

        ## 상관관계 히트맵
        st.write('상관계수 히트맵')
        corr_= data.drop(columns = ['Unnamed: 0', 'year', 'HE_glu', 'HE_HbA1c'])
        corr = corr_.corr()
    
        # 히트맵 그리기
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        st.pyplot(plt)
        
    with con4:
        st.subheader('데이터 셋')
        st.write(' ')
        with st.expander("데이터 셋 개수",expanded=False):
            st.write('''
                    #### 2011 ~ 2021 국건영 데이터
                    - Train: 10396 명
                    - Test: 2599 명
                    #### 2022년도 국건영 데이터
                    - Validation: 4969 명
                            ''')
            
        st.subheader('타겟 컬럼')
        st.write(' ')
        with st.expander("당뇨 여부 컬럼",expanded=False):
            st.write('''
                    타겟 컬럼으로 'HE_DM_HbA1c'를 사용하려 했으나, 국민건강영양조사의 기수에 따라 기준이 달라지는 점을 고려하여,
                    국민건강영양조사 데이터를 기반으로 한 당뇨병 진단 기준에 따라 값을 나누어 새로운 컬럼을 추가하고 이를 타겟 컬럼으로 사용하기로 하였습니다. 
                     
                    - **당뇨여부: 공복혈당('HE_glu') 126 이상, 당화혈색소('HE_HbA1c') 6.5%이상은 당뇨로 판단**
                            ''')
            
        st.subheader('데이터 연령대')
        st.write(' ')
        with st.expander("당뇨 여부 컬럼",expanded=False):
            st.write('''
                    - **10세 ~ 80세**
                    
                    원래 사용하려던 타겟 컬럼인 'HE_DM_HbA1c'의 당뇨병 유병 여부는 19세 이상의 데이터로 제한되었으나, 
                    새로 추가된 '당뇨 여부' 칼럼은 10세부터 시작하는 다양한 연령대의 데이터를 포함하여 학습할 수 있습니다.
                    ''') 
                     
        st.subheader('1차적으로 뽑은 22개의 컬럼')
        st.write(' ')
        with st.expander("총 22개 컬럼",expanded=False):
            st.write('''
                    'age'- 나이   /'sex'- 성별   /'HE_chol'- 총 콜레스테롤   /'HE_wc'- 허리둘레   /'HE_Ucrea'- 요크레아티닌   /'HE_BMI'- 체질량 지수   /\
                    'HE_TG'- 중성지방   /'HE_HDL_st2'- HDL 콜레스테롤   /'HE_dbp'- 이완기 혈압   /'HE_sbp'- 수축기 혈압   /'HE_crea'- 혈중 크레아티닌   /\
                    'HE_wt'- 체중   /'HE_ast'- 간장질환 AST(SGOT)   /'HE_alt'- 간장질환 ALT(SGPT)   /'HE_HP'- 고혈압 유병여부(19세이상)   /'educ'- 교육수준(학력)   /'HE_Upro'- 요단백   /\
                    'BP1'- 평소 스트레스 인지 정도   /'HE_fh'- 만성질환 의사진단 가족력 여부   /'BS3_2'- 하루평균 일반담배 흡연량(성인)   /'pa_aerobic'- 유산소 신체활동 실천율   /'DE1_dg'- 당뇨병 의사진단 여부
                            ''')

        st.subheader('최종 칼럼 설명')
        st.write(' ')
        with st.expander("총 8개의 컬럼",expanded=False):
            st.write('''
                     ##### 1차로 선정한 22개 컬럼에서 전진 선택법을 사용하여 'HE_glu' 공복혈당을 기준으로 컬럼을 추출한 후, 
                     ##### 자기상관계수가 높은 컬럼과 당뇨병 발병과 관련성이 없는 컬럼을 제거하여 최종적으로 총 8개의 컬럼을 선정하였습니다.
                    - **나이 'age'**: 나이가 증가함에 따른 변화를 확인하려고 칼럼에 넣었습니다.
                    - **성별 'sex'**: 남,녀 성별에 따른 변화를 확인하려고 칼럼에 넣었습니다.
                    - **허리 둘레 'HE_wc'**: 허리 둘레는 복부 비만을 나타내는 지표로, 내장지방의 양을 간접적으로 나타냅니다. 복부 비만은 인슐린 저항성과 밀접한 관계가 있습니다.
                    - **요단백 'HE_Upro'**: 요단백은 소변에서 단백질의 양을 측정한 것으로, 신장 기능의 지표로 사용됩니다.
                    - **혈중 크레아티닌 'HE_crea'**: 혈중 크레아티닌 수치는 신장 기능을 평가하는 데 사용됩니다. 신장 기능이 저하되면 혈중 크레아티닌 수치가 증가합니다.
                    - **간장질환 ALT 'HE_alt'**: ALT(알라닌 아미노전이효소)는 간의 기능을 평가하는 지표입니다. 간의 손상이나 염증이 있으면 ALT 수치가 상승합니다.
                    - **중성지방 'HE_TG'**: 중성지방은 혈액 내의 지방을 나타내며, 혈중 중성지방 농도는 대사 증후군의 일부로 당뇨병과 연관될 수 있습니다.
                    - **총 콜레스테롤  'HE_chol'**: 총 콜레스테롤은 혈액 내에서 운반되는 콜레스테롤의 총량을 측정한 것입니다. 이는 LDL 콜레스테롤, HDL 콜레스테롤, 그리고 중성지방을 포함합니다.
                        
                        저밀도 지단백질 콜레스테롤 (LDL-C): 종종 "나쁜" 콜레스테롤이라고 불리며, 동맥 벽에 콜레스테롤을 축적시키는 역할을 합니다.

                        고밀도 지단백질 콜레스테롤 (HDL-C): "좋은" 콜레스테롤이라고 불리며, 혈중 콜레스테롤을 간으로 운반하여 배출하도록 돕습니다.

                        중성지방 (Triglycerides): 혈중에서 콜레스테롤과 함께 존재하며, 체내 에너지원으로 사용됩니다.
                        ''')
        

    
with tab3: # 당뇨 현황 및 예측 프로그램 필요성 시각화
    st.header(":pill:  당뇨 현황")
    st.write(' ')
    st.write("**2011~2021년 총 10년치의 국민건강영양조사 데이터에 당뇨 기준을 적용해 시각화 한 자료입니다.**")
    st.markdown("___")

    con1, con2= st.columns([0.4, 0.6])
    with con1:
        # 당뇨여부별 카운트 계산
        count_us = data['당뇨여부'].value_counts().sort_index()

        # 라벨 및 값 설정
        labels = ['정상(0)', '당뇨(1)']
        values = count_us

        # Plotly 파이 차트 생성
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,  # 도넛 차트로 변경하고 싶다면 이 값을 설정합니다.
            textinfo='label+percent',  # 라벨과 퍼센트 정보 표시
            insidetextorientation='horizontal',  # 텍스트 방향 설정
            marker=dict(colors=['#66c2a5', '#fc8d62']),  # 색상 설정
            showlegend=True  # 범례 표시
        )])

        # 그래프 레이아웃 설정
        fig.update_layout(
            title_text='당뇨 비율',
            font=dict(family="Nanum Gothic", size=14),  # 한글 폰트 설정
            annotations=[dict(text='당뇨 비율', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        st.plotly_chart(fig)  # Plotly 그래프 출력

    with con2:
        count_by_year = data[data['당뇨여부'] == 1.0].groupby('year').size().reset_index(name='Count')

        # 그래프 그리기
        fig = go.Figure()

        fig = px.bar(count_by_year, x='year', y='Count',
                title='연도별 당뇨 발병 추이',
                labels={'year': '년도', 'Count': '당뇨 발병 수'},
                color='Count',
                color_continuous_scale='sunset')

        # 그래프 레이아웃 설정
        fig.update_layout(
            title='연도별 당뇨 발병 추이',
            xaxis_title='년도',
            yaxis_title='당뇨 발병 여부',
            xaxis=dict(tickmode='linear'),
            yaxis=dict(title='발병 환자 수(명)'),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig)  # Plotly 그래프 출력