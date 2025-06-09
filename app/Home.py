import streamlit as st
import sys
sys.path.append(".")

st.set_page_config(
    page_title="Streamlit Home Page"
)

st.sidebar.success("Select a page above.")
st.markdown(open("README.md").read())
