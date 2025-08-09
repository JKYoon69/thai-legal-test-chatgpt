import streamlit as st
from parser_core import parser

st.set_page_config(page_title="Thai Law Parser", layout="wide")

st.title("📜 Thai Law Parser (Test)")

uploaded_file = st.file_uploader("태국어 법률 문서 업로드", type=["txt"])

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    st.subheader("원문 미리보기")
    st.text(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)

    if st.button("파싱 실행"):
        st.info("여기에 parser.parse_document 호출 예정")
