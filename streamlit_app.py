# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import openai
from sklearn.linear_model import LinearRegression

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="AI Dietary Diversity Assistant", layout="wide")
st.title("AI Dietary Diversity Analysis Assistant")

# --- OPENAI CONFIG ---
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# --- UPLOAD DATA ---
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, Stata, or SPSS files", type=['csv','xlsx','xls','dta','sav']
)

if uploaded_file:
    # Load the file
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

    # --- SELECT ANALYSIS TYPE ---
    st.sidebar.header("Select Analysis")
    analysis_type = st.sidebar.selectbox(
        "What do you want to do with your data?",
        ["Descriptive statistics (mean, median, std)",
         "Compare variables by groups",
         "Predict a variable",
         "Generate narrative report"]
    )

    # --- SELECT COLUMNS ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    selected_numeric = []
    selected_categorical = []

    if analysis_type != "Predict a variable":
        if numeric_cols:
            selected_numeric = st.sidebar.multiselect("Select numeric columns", numeric_cols)
        if categorical_cols:
            selected_categorical = st.sidebar.multiselect("Select categorical columns", categorical_cols)

    # --- PERFORM ANALYSIS ---
    st.subheader("Analysis Results")

    if analysis_type == "Descriptive statistics (mean, median, std)":
        if selected_numeric:
            desc = df[selected_numeric].describe().T
            st.dataframe(desc)
        else:
            st.warning("Select numeric columns to compute statistics")

    elif analysis_type == "Compare variables by groups":
        if selected_numeric and selected_categorical:
            for cat in selected_categorical:
                st.markdown(f"### Comparison by {cat}")
                for num in selected_numeric:
                    grouped = df.groupby(cat)[num].agg(['mean','median','std','count'])
                    st.dataframe(grouped)
                    # Plot
                    fig, ax = plt.subplots()
                    grouped['mean'].plot(kind='bar', ax=ax)
                    ax.set_ylabel(f"Mean {num}")
                    ax.set_xlabel(cat)
                    ax.set_title(f"Mean {num} by {cat}")
                    st.pyplot(fig)
        else:
            st.warning("Select numeric and categorical columns for comparison")

    elif analysis_type == "Predict a variable":
        if numeric_cols:
            target = st.selectbox("Select target variable", numeric_cols)
            predictors = [col for col in numeric_cols if col != target]
            selected_predictors = st.multiselect("Select predictors", predictors)
            if selected_predictors:
                X = df[selected_predictors].fillna(df[selected_predictors].mean())
                y = df[target].fillna(df[target].mean())
                model = LinearRegression()
                model.fit(X, y)
                st.success("Model trained successfully")
                st.write("Predicted values for first 5 rows:")
                st.dataframe(pd.DataFrame(model.predict(X), columns=[f"Predicted {target}"]).head())
        else:
            st.warning("No numeric columns available for prediction")

    elif analysis_type == "Generate narrative report":
        st.markdown("### Narrative Report (via AI)")
        if openai.api_key:
            user_prompt = st.text_area(
                "Ask the AI to generate a narrative report",
                "Summarize the dataset and highlight key findings."
            )
            if st.button("Generate Report"):
                context = f"Dataset columns: {', '.join(df.columns)}\nData sample:\n{df.head().to_dict()}"
                prompt = f"You are a highly intelligent Data Analyst AI. {user_prompt}\nContext: {context}"
                try:
                    completion = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    answer = completion.choices[0].message.content
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
        else:
            st.warning("OpenAI API key not set. Narrative report not available.")

    # --- DOWNLOAD DATA & PDF ---
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed CSV", data=csv_bytes, file_name="processed_data.csv", mime="text/csv")

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for col in selected_numeric:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=12)
            ax.set_title(f"{col} Distribution")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            pdf.savefig(fig)
            plt.close(fig)
        for cat in selected_categorical:
            for num in selected_numeric:
                fig, ax = plt.subplots()
                df.boxplot(column=num, by=cat, ax=ax)
                ax.set_title(f"{num} by {cat}")
                pdf.savefig(fig)
                plt.close(fig)
    pdf_buffer.seek(0)
    st.download_button("Download PDF report", data=pdf_buffer, file_name="report.pdf", mime="application/pdf")

else:
    st.info("Upload a dataset to start analysis.")
