import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import openai
import numpy as np
from sklearn.linear_model import LinearRegression

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Dietary Diversity Assistant", layout="wide")
st.title("AI Dietary Diversity Analysis Assistant")

# --- OPENAI CONFIG ---
# You need to set your API key in Streamlit secrets or as environment variable
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# --- UPLOAD DATA ---
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, or Stata/SPSS files", 
    type=['csv','xlsx','xls','dta','sav']
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls','xlsx')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.dta'):
            df = pd.read_stata(uploaded_file)
        elif uploaded_file.name.endswith('.sav'):
            import pyreadstat
            df, meta = pyreadstat.read_sav(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.success("File loaded successfully")
    st.dataframe(df.head())

    # --- SELECT COLUMNS ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    st.sidebar.header("Select Columns")
    dds_col = st.sidebar.selectbox("Select DDS column", numeric_cols)
    cat_cols = st.sidebar.multiselect("Select categorical columns for breakdown", categorical_cols)

    # --- SUMMARY STATISTICS ---
    st.subheader("Summary Statistics")
    mean_dds = df[dds_col].mean()
    std_dds = df[dds_col].std()
    st.metric("Overall mean DDS", f"{mean_dds:.2f}")
    st.metric("Overall DDS SD", f"{std_dds:.2f}")

    if cat_cols:
        st.subheader("Grouped Summary")
        grouped = df.groupby(cat_cols)[dds_col].agg(['mean','std','count']).reset_index()
        st.dataframe(grouped)

        # --- PLOT GROUPED DATA ---
        for cat in cat_cols[:3]:  # first 3 categorical variables
            st.markdown(f"**DDS by {cat}**")
            fig, ax = plt.subplots()
            df.groupby(cat)[dds_col].mean().plot(kind='bar', ax=ax)
            ax.set_ylabel("Mean DDS")
            ax.set_xlabel(cat)
            st.pyplot(fig)

    # --- DOWNLOAD DATA & PDF ---
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed CSV", data=csv_bytes, file_name="processed_dds.csv", mime="text/csv")

    # PDF report
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots()
        ax.hist(df[dds_col].dropna(), bins=12)
        ax.set_title("DDS Distribution")
        ax.set_xlabel("DDS")
        ax.set_ylabel("Frequency")
        pdf.savefig(fig)
        plt.close(fig)

        for cat in cat_cols[:3]:
            fig, ax = plt.subplots()
            df.boxplot(column=dds_col, by=cat, ax=ax)
            ax.set_title(f"DDS by {cat}")
            pdf.savefig(fig)
            plt.close(fig)

    pdf_buffer.seek(0)
    st.download_button("Download PDF report", data=pdf_buffer, file_name="dds_report.pdf", mime="application/pdf")

    # --- AI CONVERSATIONAL INTERFACE ---
    st.subheader("Ask AI questions about your data")
    user_question = st.text_input("Ask a question (e.g., 'Which region has highest DDS?')")

    if user_question:
        # Prepare context
        context = f"Dataset columns: {', '.join(df.columns)}. Show summary statistics if needed.\nData sample:\n{df.head().to_dict()}"
        prompt = f"Answer the following question based on the dataset: {user_question}\nContext: {context}"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user","content": prompt}],
                temperature=0
            )
            answer = response['choices'][0]['message']['content']
            st.markdown(f"**AI Answer:** {answer}")
        except Exception as e:
            st.error(f"OpenAI API error: {e}")

    # --- SIMPLE PREDICTIVE EXAMPLE ---
    st.subheader("Predict DDS from numeric predictors")
    numeric_predictors = [col for col in numeric_cols if col != dds_col]
    if numeric_predictors:
        selected_predictors = st.multiselect("Select numeric predictors", numeric_predictors)
        if selected_predictors:
            X = df[selected_predictors].fillna(df[selected_predictors].mean())
            y = df[dds_col]
            model = LinearRegression()
            model.fit(X, y)
            st.success("Model trained successfully")
            st.write("Predicted DDS for first 5 rows:")
            st.dataframe(pd.DataFrame(model.predict(X), columns=["Predicted DDS"]).head())

st.info("Upload a dataset to start analysis.")
