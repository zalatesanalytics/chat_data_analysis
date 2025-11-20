# streamlit_app.py
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

    # --- SELECT MULTIPLE ANALYSIS TYPES ---
    st.sidebar.header("Select Analysis Actions")
    actions = st.sidebar.multiselect(
        "Choose one or more actions to perform:",
        [
            "Descriptive statistics (mean, median, std)",
            "Compare variables by groups",
            "Predict a variable",
            "Generate narrative report"
        ]
    )

    # --- SELECT COLUMNS ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    selected_numeric = []
    selected_categorical = []

    if "Descriptive statistics (mean, median, std)" in actions or "Compare variables by groups" in actions:
        if numeric_cols:
            selected_numeric = st.sidebar.multiselect("Select numeric columns", numeric_cols)
        if categorical_cols:
            selected_categorical = st.sidebar.multiselect("Select categorical columns", categorical_cols)

    if "Predict a variable" in actions and numeric_cols:
        target = st.sidebar.selectbox("Select target variable for prediction", numeric_cols)
        predictors = [col for col in numeric_cols if col != target]
        selected_predictors = st.sidebar.multiselect("Select numeric predictors", predictors)

    # --- PERFORM ANALYSES ---
    st.subheader("Analysis Results")
    narrative_reports = []

    # Descriptive statistics
    if "Descriptive statistics (mean, median, std)" in actions:
        if selected_numeric:
            desc = df[selected_numeric].describe().T
            st.markdown("### Descriptive Statistics")
            st.dataframe(desc)
            narrative_reports.append(f"Descriptive statistics for selected numeric columns:\n{desc.to_string()}")
        else:
            st.warning("Select numeric columns to compute statistics")

    # Comparison by groups
    if "Compare variables by groups" in actions:
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
                    narrative_reports.append(f"Comparison of {num} by {cat}:\n{grouped.to_string()}")
        else:
            st.warning("Select numeric and categorical columns for comparison")

    # Prediction
    if "Predict a variable" in actions:
        if selected_predictors:
            X = df[selected_predictors].fillna(df[selected_predictors].mean())
            y = df[target].fillna(df[target].mean())
            model = LinearRegression()
            model.fit(X, y)
            st.markdown(f"### Prediction for {target}")
            st.success("Model trained successfully")
            predicted = pd.DataFrame(model.predict(X), columns=[f"Predicted {target}"])
            st.dataframe(predicted.head())
            narrative_reports.append(f"Predicted {target} values based on selected predictors:\n{predicted.head().to_string()}")
        else:
            st.warning("Select predictors to train prediction model")

    # AI Narrative report
    if "Generate narrative report" in actions:
        st.markdown("### Narrative Report (via AI)")
        if openai.api_key:
            user_prompt = st.text_area(
                "Ask the AI to generate a narrative report based on your selected analyses",
                "Summarize the dataset, describe findings from all selected actions, and highlight key insights."
            )
            if st.button("Generate AI Report"):
                context = f"Dataset columns: {', '.join(df.columns)}\nData sample:\n{df.head().to_dict()}"
                # Combine previous narratives
                previous_report = "\n\n".join(narrative_reports) if narrative_reports else ""
                prompt = f"You are a highly intelligent Data Analyst AI. {user_prompt}\n\nPrevious analysis summaries:\n{previous_report}\nContext: {context}"
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

    # --- DOWNLOAD DATA & PDF REPORT ---
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed CSV", data=csv_bytes, file_name="processed_data.csv", mime="text/csv")

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Numeric distributions
        for col in selected_numeric:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=12)
            ax.set_title(f"{col} Distribution")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            pdf.savefig(fig)
            plt.close(fig)
        # Boxplots
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
