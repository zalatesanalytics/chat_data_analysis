import streamlit as st
import pandas as pd

st.title("AI Data Analysis Assistant")

uploaded = st.file_uploader("Upload your dataset (Excel, CSV, Stata, SPSS, PDF)", type=["csv", "xlsx", "xls", "dta", "sav", "pdf"])

if uploaded:
    st.success("File uploaded!")
