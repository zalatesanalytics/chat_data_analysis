# streamlit_app.py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

import openai

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="AI Food & Youth Analysis Assistant", layout="wide")
st.title("AI Food Security, Nutrition & Youth Development Analysis Assistant")

# ---------------- OPENAI CONFIG -------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# ==================================================
# SCORING HELPERS: HFIAS / HDDS / WDDS
# ==================================================

def categorize_hfias(score):
    """Categorize HFIAS score into standard severity groups."""
    if pd.isna(score):
        return np.nan
    if score <= 1:
        return "Food secure"
    elif 2 <= score <= 8:
        return "Mildly food insecure"
    elif 9 <= score <= 16:
        return "Moderately food insecure"
    else:
        return "Severely food insecure"


def compute_hfias_scores(
    df: pd.DataFrame,
    question_cols=None,
    score_col: str = "hfias_score",
    category_col: str = "hfias_category",
):
    """
    Compute HFIAS total score and severity category.
    """
    if question_cols is None:
        question_cols = [c for c in df.columns if c.lower().startswith("hfias_q")]

    if not question_cols:
        return df  # nothing to do

    df[score_col] = df[question_cols].sum(axis=1)
    df[category_col] = df[score_col].apply(categorize_hfias)
    return df


def compute_dietary_diversity_score(
    df: pd.DataFrame,
    food_group_cols,
    score_col: str = "dds_score",
    max_score: int | None = None,
):
    """
    Compute dietary diversity score (HDDS or WDDS) as the sum of binary food group indicators.
    """
    if not food_group_cols:
        return df

    # Treat any positive value as 1 (consumed)
    binary = df[food_group_cols].gt(0).astype(int)
    df[score_col] = binary.sum(axis=1)

    if max_score is not None:
        df[score_col] = df[score_col].clip(upper=max_score)

    return df


# ==================================================
# DUMMY DATASET GENERATORS (NOW USING THE SCORERS)
# ==================================================

def create_dummy_hfias(n=150):
    np.random.seed(42)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "sex_head": np.random.choice(["Male", "Female"], n),
        "hfias_q1": np.random.randint(0, 4, n),
        "hfias_q2": np.random.randint(0, 4, n),
        "hfias_q3": np.random.randint(0, 4, n),
        "hfias_q4": np.random.randint(0, 4, n),
        "hfias_q5": np.random.randint(0, 4, n),
        "hfias_q6": np.random.randint(0, 4, n),
        "hfias_q7": np.random.randint(0, 4, n),
        "hfias_q8": np.random.randint(0, 4, n),
        "hfias_q9": np.random.randint(0, 4, n),
    })

    # Use explicit HFIAS calculator
    question_cols = [f"hfias_q{i}" for i in range(1, 10)]
    df = compute_hfias_scores(df, question_cols=question_cols,
                              score_col="hfias_score",
                              category_col="hfias_category")
    return df


def create_dummy_wdds(n=150):
    np.random.seed(43)
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "age": np.random.randint(15, 49, n),
        "sex": "Female",
        "region": np.random.choice(["Urban", "Rural"], n),
        "education": np.random.choice(["None", "Primary", "Secondary", "Tertiary"], n),
    })

    # 9 WDDS food group indicators (binary)
    for i in range(1, 10):
        df[f"wdds_fg{i}"] = np.random.binomial(1, 0.6, n)

    # Use explicit WDDS calculator
    food_groups = [f"wdds_fg{i}" for i in range(1, 10)]
    df = compute_dietary_diversity_score(
        df, food_group_cols=food_groups, score_col="wdds", max_score=9
    )
    return df


def create_dummy_child_malnutrition(n=150):
    np.random.seed(44)
    df = pd.DataFrame({
        "child_id": range(1, n + 1),
        "age_months": np.random.randint(6, 60, n),
        "sex": np.random.choice(["Male", "Female"], n),
        "weight_kg": np.round(np.random.normal(12, 2.5, n), 1),
        "height_cm": np.round(np.random.normal(90, 8, n), 1),
        "wfh_zscore": np.round(np.random.normal(-0.5, 1.2, n), 2)
    })
    df["malnutrition_type"] = pd.cut(
        df["wfh_zscore"],
        bins=[-10, -3, -2, 100],
        labels=["Severe wasting", "Moderate wasting", "Normal"]
    )
    return df


def create_dummy_consumption_production(n=150):
    np.random.seed(45)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "monthly_food_expense": np.round(np.random.normal(120, 40, n), 2),
        "monthly_income": np.round(np.random.normal(300, 100, n), 2),
        "produces_own_food": np.random.choice([0, 1], n),
        "livestock_count": np.random.poisson(3, n)
    })

    # 12 HDDS food group indicators (binary)
    for i in range(1, 13):
        df[f"hdds_fg{i}"] = np.random.binomial(1, 0.7, n)

    # Compute explicit HDDS score
    food_groups = [f"hdds_fg{i}" for i in range(1, 13)]
    df = compute_dietary_diversity_score(
        df, food_group_cols=food_groups, score_col="hdds", max_score=12
    )
    return df


def create_dummy_youth_decision(n=150):
    np.random.seed(46)
    df = pd.DataFrame({
        "youth_id": range(1, n + 1),
        "age": np.random.randint(15, 29, n),
        "sex": np.random.choice(["Male", "Female"], n),
        "education": np.random.choice(["None", "Primary", "Secondary", "Tertiary"], n),
        "employment_status": np.random.choice(
            ["Unemployed", "Employed", "Self-employed", "Student"], n
        ),
        "decision_score": np.random.randint(1, 6, n),  # 1–5 Likert
        "agency_score": np.random.randint(1, 6, n),
        "aspiration_score": np.random.randint(1, 6, n),
        "participation_score": np.random.randint(1, 6, n)
    })
    df["received_training"] = np.random.choice([0, 1], n)
    return df


def create_dummy_integrated(n=200):
    np.random.seed(47)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "sex_head": np.random.choice(["Male", "Female"], n),
        "monthly_income": np.round(np.random.normal(320, 120, n), 2),
        "monthly_food_expense": np.round(np.random.normal(130, 45, n), 2),
        "youth_in_household": np.random.randint(0, 4, n),
        "youth_decision_score": np.random.randint(1, 6, n),
        "youth_agency_score": np.random.randint(1, 6, n),
    })

    # HFIAS questions
    for i in range(1, 10):
        df[f"hfias_q{i}"] = np.random.randint(0, 4, n)
    df = compute_hfias_scores(
        df, question_cols=[f"hfias_q{i}" for i in range(1, 10)],
        score_col="hfias_score", category_col="hfias_category"
    )

    # HDDS groups (12)
    for i in range(1, 13):
        df[f"hdds_fg{i}"] = np.random.binomial(1, 0.7, n)
    df = compute_dietary_diversity_score(
        df, food_group_cols=[f"hdds_fg{i}" for i in range(1, 13)],
        score_col="hdds", max_score=12
    )

    # WDDS groups (9)
    for i in range(1, 10):
        df[f"wdds_fg{i}"] = np.random.binomial(1, 0.6, n)
    df = compute_dietary_diversity_score(
        df, food_group_cols=[f"wdds_fg{i}" for i in range(1, 10)],
        score_col="wdds", max_score=9
    )

    return df


# ==================================================
# SIDEBAR: CHOOSE SAMPLE OR UPLOAD
# ==================================================
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use sample (dummy) dataset", "Upload your own dataset"]
)

df = None
dataset_label = None

if data_source == "Use sample (dummy) dataset":
    sample_choice = st.sidebar.selectbox(
        "Select sample dataset",
        [
            "HFIAS household food insecurity",
            "Women’s dietary diversity (WDDS)",
            "Child malnutrition / anthropometry",
            "Food consumption & production",
            "Youth development & decision-making",
            "Integrated multi-topic dataset"
        ]
    )

    if sample_choice == "HFIAS household food insecurity":
        df = create_dummy_hfias()
        dataset_label = "Dummy HFIAS dataset"
    elif sample_choice == "Women’s dietary diversity (WDDS)":
        df = create_dummy_wdds()
        dataset_label = "Dummy WDDS dataset"
    elif sample_choice == "Child malnutrition / anthropometry":
        df = create_dummy_child_malnutrition()
        dataset_label = "Dummy child malnutrition dataset"
    elif sample_choice == "Food consumption & production":
        df = create_dummy_consumption_production()
        dataset_label = "Dummy consumption & production dataset"
    elif sample_choice == "Youth development & decision-making":
        df = create_dummy_youth_decision()
        dataset_label = "Dummy youth decision-making dataset"
    elif sample_choice == "Integrated multi-topic dataset":
        df = create_dummy_integrated()
        dataset_label = "Dummy integrated dataset"

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, Excel, JSON, TSV, Stata, SPSS, or PDF file",
        type=["csv", "xlsx", "xls", "json", "tsv", "txt", "dta", "sav", "pdf"]
    )
    if uploaded_file is not None:
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            elif name.endswith((".tsv", ".txt")):
                df = pd.read_csv(uploaded_file, sep="\t")
            elif name.endswith(".dta"):
                df = pd.read_stata(uploaded_file)
            elif name.endswith(".sav"):
                import pyreadstat
                df, meta = pyreadstat.read_sav(uploaded_file)
            elif name.endswith(".pdf"):
                # Basic PDF support: convert pages to text
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(uploaded_file)
                    pages = []
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        pages.append({"page": i + 1, "text": text})
                    df = pd.DataFrame(pages)
                    dataset_label = "PDF text (page-level) dataset"
                    st.info(
                        "PDF loaded as page-level text. "
                        "You can generate AI narrative from this text, but numeric analysis may be limited."
                    )
                except ImportError:
                    st.error(
                        "PDF support requires the 'pypdf' package. "
                        "Add 'pypdf' to your requirements.txt to enable PDF ingestion."
                    )
                    df = None
            else:
                st.error("Unsupported file format.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

if df is None:
    st.info("Select a sample dataset or upload your own file to begin.")
    st.stop()

# Show basic info
st.success(f"Dataset loaded: {dataset_label or 'Uploaded dataset'}")
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

# ==================================================
# ANALYSIS MODE: AI AUTO vs SCRIPT
# ==================================================
st.sidebar.header("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose analysis mode",
    ["AI auto-detect analysis", "Use custom analysis script"]
)

narrative_chunks = []  # for AI narrative later

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ==================================================
# CUSTOM SCRIPT MODE (STUB + BASIC ANALYSIS)
# ==================================================
if analysis_mode == "Use custom analysis script":
    st.subheader("Custom Script–Driven Analysis")

    script_type = st.selectbox(
        "Type of script you want to reference",
        ["Stata (.do)", "SPSS (.sps)", "Python code (text)"]
    )

    script_text = st.text_area(
        "Paste your analysis script here (for now, used as guidance for AI + summaries)",
        height=200,
        placeholder="Paste your .do / .sps / Python logic here..."
    )

    st.markdown("### Basic descriptive analysis based on your dataset")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)
        narrative_chunks.append("Descriptive statistics (custom script mode):\n" + desc.to_string())
    else:
        st.warning("No numeric columns detected for descriptive statistics.")

    st.info(
        "This version summarizes your data and can use the script text "
        "as context for the AI narrative. You can extend it later to parse "
        "and mirror Stata/SPSS logic."
    )

else:
    # ==================================================
    # AI AUTO-DETECT ANALYSIS MODE
    # ==================================================
    st.subheader("AI Auto-Detected Analysis")

    # -------- HFIAS --------
    if "hfias_score" in df.columns:
        st.markdown("### HFIAS Summary")
        hfias = df["hfias_score"]
        st.write("HFIAS score descriptive statistics:")
        st.dataframe(hfias.describe().to_frame().T)

        cat_col = "hfias_category" if "hfias_category" in df.columns else None
        if not cat_col:
            # If category not present (e.g., user data), try to compute using helper
            df = compute_hfias_scores(df, question_cols=None,
                                      score_col="hfias_score",
                                      category_col="hfias_category")
            cat_col = "hfias_category"

        st.write("HFIAS severity distribution:")
        hfias_counts = df[cat_col].value_counts(dropna=False)
        st.bar_chart(hfias_counts)

        narrative_chunks.append(
            "HFIAS distribution:\n" + hfias_counts.to_string()
        )

    # -------- HDDS / WDDS --------
    for col in ["hdds", "wdds"]:
        if col in df.columns:
            st.markdown(f"### {col.upper()} summary")
            st.write(df[col].describe())
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=10)
            ax.set_title(f"Distribution of {col.upper()}")
            ax.set_xlabel(col.upper())
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            narrative_chunks.append(
                f"{col.upper()} distribution summary:\n{df[col].describe().to_string()}"
            )

    # -------- CHILD ANTHROPOMETRY --------
    if "wfh_zscore" in df.columns:
        st.markdown("### Child Weight-for-Height Z-score")
        st.write(df["wfh_zscore"].describe())
        fig, ax = plt.subplots()
        ax.hist(df["wfh_zscore"].dropna(), bins=10)
        ax.set_title("Distribution of WFH Z-scores")
        ax.set_xlabel("WFH Z-score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        narrative_chunks.append(
            "WFH z-score distribution:\n" + df["wfh_zscore"].describe().to_string()
        )
        if "malnutrition_type" in df.columns:
            st.write("Malnutrition categories:")
            mal_counts = df["malnutrition_type"].value_counts(dropna=False)
            st.bar_chart(mal_counts)
            narrative_chunks.append(
                "Malnutrition type frequencies:\n" + mal_counts.to_string()
            )

    # -------- YOUTH DECISION-MAKING --------
    youth_score_cols = [
        c for c in df.columns
        if "decision_score" in c.lower()
        or "agency_score" in c.lower()
        or "aspiration_score" in c.lower()
        or "participation_score" in c.lower()
        or "youth_decision_score" in c.lower()
        or "youth_agency_score" in c.lower()
    ]
    if youth_score_cols:
        st.markdown("### Youth Decision-Making & Agency Scores")
        st.dataframe(df[youth_score_cols].describe().T)
        narrative_chunks.append(
            "Youth empowerment/decision-related scores:\n"
            + df[youth_score_cols].describe().T.to_string()
        )
        if "sex" in df.columns:
            st.markdown("#### Scores by sex")
            for col in youth_score_cols:
                grouped = df.groupby("sex")[col].mean()
                st.write(f"Mean {col} by sex:")
                st.dataframe(grouped.to_frame())
                narrative_chunks.append(
                    f"Mean {col} by sex:\n{grouped.to_string()}"
                )

    # -------- GENERAL DESCRIPTIVES --------
    st.markdown("### General Descriptive Statistics")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)
        narrative_chunks.append("Overall numeric descriptives:\n" + desc.to_string())
    if categorical_cols:
        st.markdown("### Key Categorical Distributions")
        for col in categorical_cols[:10]:  # limit to first 10
            st.write(f"**{col}** value counts:")
            vc = df[col].value_counts(dropna=False)
            st.dataframe(vc.to_frame("count"))
            narrative_chunks.append(f"Value counts for {col}:\n{vc.to_string()}")


# ==================================================
# AI NARRATIVE REPORT
# ==================================================
st.markdown("---")
st.subheader("AI Narrative Report")

if not openai.api_key:
    st.warning("OpenAI API key not set in Streamlit secrets. Narrative report not available.")
else:
    user_prompt = st.text_area(
        "Optional: refine what you want the AI to focus on",
        value=(
            "Summarize the dataset, describe key findings on food security, "
            "dietary diversity, youth decision-making, and income/consumption where applicable. "
            "Highlight any gender or regional differences and potential program implications."
        )
    )

    if st.button("Generate AI Narrative"):
        context_sample = df.head(10).to_dict()
        combined_narrative = "\n\n".join(narrative_chunks) if narrative_chunks else "No prior summaries."

        prompt = (
            "You are an expert data analyst working on food security, nutrition, and youth development.\n"
            "Write a clear, non-technical narrative (1–2 pages) summarizing the key insights "
            "from the analysis below, and suggest 3–5 program or policy implications.\n\n"
            f"USER FOCUS: {user_prompt}\n\n"
            "ANALYSIS SUMMARIES:\n"
            f"{combined_narrative}\n\n"
            "DATA PREVIEW (first 10 rows as dict):\n"
            f"{context_sample}\n"
        )

        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            ai_text = completion.choices[0].message.content
            st.markdown(ai_text)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")

# ==================================================
# DOWNLOADS: CSV + PDF REPORT
# ==================================================
st.markdown("---")
st.subheader("Downloads")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download current dataset as CSV",
    data=csv_bytes,
    file_name="dataset_processed.csv",
    mime="text/csv"
)

pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer) as pdf:
    # Numeric distributions
    for col in numeric_cols[:12]:  # limit to 12 columns for PDF
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=12)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        pdf.savefig(fig)
        plt.close(fig)
pdf_buffer.seek(0)

st.download_button(
    "Download basic PDF chart report",
    data=pdf_buffer,
    file_name="basic_report.pdf",
    mime="application/pdf"
)
