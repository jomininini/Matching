import os
import pandas as pd
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import base64
from io import BytesIO
from refine import refine_input
from company_evaluator import evaluate_company

openai_api_key = os.getenv('OPENAI_API_KEY')

# 配置Streamlit页面
st.set_page_config(layout="wide", page_icon="💬", page_title="HKSTP🤖")

# 页面标题
st.markdown(
    f"""
    <h1 style='text-align: center;'> Matching Tools-Companies, Solutions & Investors</h1>
    """,
    unsafe_allow_html=True,
)

# 侧边栏输入
matching_option = st.sidebar.selectbox("Matching", ["Company Matching", "Funds Matching", "Solution Matching"])

# 用户输入框
raw_input = st.text_area("Please input your statement:", key='raw_input_area')

# 调用 refine_input 函数并在输入后按 Enter 键提交
if st.button('Refine Input'):
    refined_input_text = refine_input(matching_option, raw_input)
    st.session_state['refined_input_text'] = refined_input_text
else:
    if 'refined_input_text' not in st.session_state:
        st.session_state['refined_input_text'] = "The refined input will show here"

# 显示 refined_input_text 框架并允许编辑
refined_input_text = st.text_area("Refined Input (editable):", st.session_state['refined_input_text'], height=100)

# 根据匹配选项加载不同的索引和数据框
if matching_option == "Company Matching":
    db = FAISS.load_local("index_hkstp_new", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    df = pd.read_csv("hkstp_company_directory.csv")
    default_columns = ['name_EN', 'introduction_EN', 'product_EN', 'website']
elif matching_option == "Solution Matching":
    db = FAISS.load_local("index_solutions", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    df = pd.read_csv("2_solutions.csv")
    default_columns = ['Title', 'web_content', 'Link', 'Institute']

# 初始化检索器
retriever = db.as_retriever(search_kwargs={"k": len(df)})  # 返回所有结果

background = st.sidebar.text_area("Backgrounds", height=200)
# 设置 business_needs 的默认值为 refined_input_text，并允许编辑
business_needs = st.sidebar.text_area("Business Needs", value=refined_input_text, height=200)

# 确保 background 和 business_needs 不是 None
#background = background if background is not None else ""
#business_needs = business_needs if business_needs is not None else ""

# 默认的 prompt_statement
default_prompt = (
    "you are a professional industry expert in innovation, technology, investment, consulting. " +
    "please consider the background: " + background + "\n\n" +
    "and the business needs: to find " + business_needs +
    " tell if the following company is in the target, if it matches the business needs and give the reason." + "\n\n" +
    "company info: "
)

prompt = st.sidebar.text_area("Prompt", value=default_prompt, height=300)

# 提交 refined input 进行匹配
if st.button('Submit Refined Input for Matching'):
    row = []
    docs = retriever.get_relevant_documents(refined_input_text)
    
    if len(docs) == 0:
        st.write("No relevant documents found.")
    else:
        for doc in docs:
            num = doc.metadata['row']
            row.append(num)
        
        result = df.loc[row]
        
        st.session_state['matching_result'] = result
        #st.write("Here is the result DataFrame:")
        #st.write(result)

# 选择要显示的列
all_columns = df.columns.tolist()
Top_key = st.sidebar.number_input("Please input the top number:", min_value=1, key='top_number_input')
columns_to_display = st.sidebar.multiselect("Select columns to display:", all_columns, default=default_columns)

# 显示匹配结果
if 'matching_result' in st.session_state:
    st.write("Here is the result DataFrame:")
    st.write(st.session_state['matching_result'])

# 过滤并显示 Top K 结果
if st.button('Filter the Dataframe'):
    if 'matching_result' in st.session_state:
        result = st.session_state['matching_result']
        simple_result = result.head(Top_key)[columns_to_display]
        st.session_state['filtered_result'] = simple_result
        st.session_state['analysis_result'] = result.head(Top_key)

# 显示过滤结果
if 'filtered_result' in st.session_state:
    st.write("Here is the filtered DataFrame:")
    st.write(st.session_state['filtered_result'])

#  Analysis按钮
if st.button('Analysis'):
    if 'analysis_result' in st.session_state:
        analysis_result = st.session_state['analysis_result']
        # 为 DataFrame 添加 'Yes/No' 和 'Reason' 列
        if 'Yes/No' not in analysis_result.columns:
            analysis_result['Yes/No'] = ''
        if 'Reason' not in analysis_result.columns:
            analysis_result['Reason'] = ''
        
        for index, row in analysis_result.iterrows():
            evaluation = evaluate_company(row, background, business_needs, prompt)
            analysis_result.at[index, 'Yes/No'] = evaluation['Yes/No']
            analysis_result.at[index, 'Reason'] = evaluation['Reason']
            
            # 更新行数据后，再显示当前行的分析结果，以列名: 内容的方式
            st.write(f"Company {index}:")
            for col in analysis_result.columns:
                if col in ['Yes/No', 'Reason']:
                    st.write(f"{col}: {evaluation[col]}")
                else:
                    st.write(f"{col}: {row[col]}")
            st.write("---")
        
        st.session_state['analyzed_result'] = analysis_result

# 显示分析结果
if 'analyzed_result' in st.session_state:
    st.write("Here is the analyzed DataFrame:")
    columns_to_display_new = columns_to_display + ['Yes/No', 'Reason']
    st.write(st.session_state['analyzed_result'][columns_to_display_new])

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        towrite = BytesIO()
        object_to_download.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

if st.button('Download Data as XLSX'):
    if 'analyzed_result' in st.session_state:
        tmp_download_link = download_link(st.session_state['analyzed_result'], 'result.xlsx', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    else:
        st.write("No data to download.")