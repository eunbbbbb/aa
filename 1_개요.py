import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 페이지 설정
st.set_page_config(layout="wide")
st.header(':pill:  당뇨병이란')
con1, con2= st.columns([0.9, 0.1])

## main ##
with con1:
    with st.container():
        tab1, tab2, tab3, tab4= st.tabs(["정의", "원인", "증상", "진단"])

    with tab1:
        st.subheader(':small_blue_diamond:  당뇨란')
        with st.expander(" ",expanded=True):
            st.write('''
                    - 체내 인슐린 생성 또는 작용의 문제로 인해 **혈당 조절이 제대로 이루어지지 않는 대사 질환**입니다.
                    - 장기적으로 다양한 합병증을 유발할 수 있어 **조기 진단과 관리가 중요**합니다.
                        ''')
        st.subheader(':small_blue_diamond:  연도별 당뇨 발병 추이')
        with st.expander("출처: 국민건강영양조사 기반의 당뇨병 관리지표 심층보고서", expanded=True):
            # Data for visualization
            # 데이터 준비
            data = {
                    '5기(2011~2012)': [12.1, 13.5, 10.8],
                    '6기(2013~2015)': [12.9, 14.6, 11.4],
                    '7기(2016~2018)': [13.9, 15.6, 12.2],
                    '8기(2019~2021)': [15.8, 18.3, 13.5]
                        }

            # 데이터프레임 생성
            df = pd.DataFrame(data, index=['전체(30세+)', '남자', '여자'])

            # 라인 플롯 생성
            fig = go.Figure()

            for gender in df.index:
                fig.add_trace(go.Scatter(
                    x=df.columns,
                    y=df.loc[gender],
                    mode='lines+markers+text',
                    name=gender,
                    text=df.loc[gender],
                    textposition='top center'  # 텍스트 위치 조정
                ))

            # 레이아웃 설정
            fig.update_layout(
                title='당뇨병 유병률 변화',
                xaxis_title='국민 건강 영양 조사 기수',
                yaxis_title='유병률 (%)',
                legend_title='성별'
            )

            # Streamlit에서 Plotly 그래프 출력
            st.plotly_chart(fig)
            st.write('''
                    - 당뇨병의 유병률은 전 세계적으로 **꾸준히 증가**하고 있으며, 특히 한국에서는 그 증가 속도가 더욱 두드러지고 있습니다.
                    - **한국 내 인구의 절반 이상이 자신의 당뇨병 여부를 인식하지 못하고 있어**, 이에 따른 건강 관리의 중요성이 더욱 강조되고 있습니다.
                    ''')


    with tab2:
        st.subheader(':small_blue_diamond:  유전적 요인')
        with st.expander("",expanded=True):
            st.markdown('''
        - 당뇨병의 발병 원인은 아직 정확히 밝혀지지 않았지만, <b>유전적 요인이 주요 원인</b>으로 고려됩니다.
        - 부모가 <b>모두 당뇨병인 경우 자녀의 발병 확률은 약 30%, 한 쪽만 당뇨병인 경우는 약 15%</b>입니다.
        - 유전적 요인이 있다고 해서 모두 당뇨병에 걸리는 것은 아니며, <b>다양한 환경적 요인이 함께 작용</b>하여 당뇨병이 발생할 수 있습니다.
    ''', unsafe_allow_html=True)
        st.subheader(':small_blue_diamond:  환경적 요인')
        with st.expander("",expanded=True):
            D1, D2 = st.columns([0.5,0.5])
            with D1:
                st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/당뇨병원인.jpg', caption='출처 : 질병관리청' )
            with D2:
                st.markdown('''
                            - **비만**
                            - **연령**
                            - **식생활**
                            - **운동부족**
                            - **스트레스**
                            - **성별**
                            - **호르몬 분비**
                            - **감염증**
                            - **약물복용**
                            - **외과적 수술**
                            ''')
    with tab3:
        st.subheader('당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다')
        st.subheader(':small_blue_diamond: 대표적인 3대 증상')
        con3, con4, con5 = st.columns([0.3, 0.3, 0.3])
        with con3:
            st.markdown('<div style="font-size: 23px;"> <b>다음(多飮)</b></div>', unsafe_allow_html=True)
            st.write('<span style="font-size: 18px;">&#8226; </span> 갈증이 심해 물을 많이 마시게 됩니다', unsafe_allow_html=True)
            st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/다음.png', caption='출처 : 대한당뇨병협회')
        with con4:
            st.markdown('<div style="font-size: 23px;"> <b>다식(多食)</b></div>', unsafe_allow_html=True)
            st.write('<span style="font-size: 18px;">&#8226; </span> 공복감이 심해 점점 더 먹으려 합니다', unsafe_allow_html=True)
            st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/다식.png', caption='출처 : 대한당뇨병협회')
        with con5:
            st.markdown('<div style="font-size: 23px;"> <b>다뇨(多尿)</b></div>', unsafe_allow_html=True)
            st.write('<span style="font-size: 18px;">&#8226; </span> 소변을 많이 보게 됩니다', unsafe_allow_html=True)
            st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/다뇨.png', caption='출처 : 대한당뇨병협회')
        con6, con7 = st.columns([0.5,0.5])
        with con6:
            st.subheader(':small_blue_diamond: 전신 증상')
            st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/전신증상.jpg', caption='출처 : 질병관리청')
        with con7:
            st.subheader(':small_blue_diamond: 기타 증상')
            st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/기타증상.jpg', caption='출처 : 질병관리청')

    with tab4:
        st.subheader('당뇨병은 혈당 검사로 진단합니다')
        st.subheader(':small_blue_diamond: 당뇨병 진단 기준')
        st.write('다음 중 1개 이상 해당 시 당뇨병으로 진단합니다')
        st.image('C:/Users/1104_6/OneDrive/바탕 화면/고은비 팀프로젝트 관련 논문/그림.png', caption='출처 : 질병관리청', use_column_width=True)
        st.subheader(':small_blue_diamond: 당뇨병 검사')
        st.write('다음과 같은 경우, 당뇨병에 대한 검사를 해보는 것이 좋습니다')
        with st.expander("",expanded=True):
            st.markdown('''
                    - 40세 이상으로 비만한 사람
                    - 가족력 가까운 친척 중에서 당뇨병이 있는 사람
                    - 자각증상 갈증, 다음, 다뇨, 다식, 피로감, 체중감소 등의 증상이 있는 사람
                    - 당뇨병이 합병되기 쉬운 질환(고혈압, 췌장염, 내분비 질환, 담석증)이 있는 사람
                    - 당뇨병 발병을 촉진하는 약물(다이아자이드계 혈압 강하제나 신경통에 쓰이는 부신피질 호르몬인 스테로이드 제품)을 사용하고 있는 사람
                        ''')