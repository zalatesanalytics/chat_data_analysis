# streamlit_app.py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

import openai
import requests  # for KoboToolbox API calls

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

def create_dummy_hfias(n=500):
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
    df = compute_hfias_scores(
        df,
        question_cols=question_cols,
        score_col="hfias_score",
        category_col="hfias_category",
    )
    return df


def create_dummy_wdds(n=500):
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


def create_dummy_child_malnutrition(n=500):
    np.random.seed(44)
    df = pd.DataFrame({
        "child_id": range(1, n + 1),
        "age_months": np.random.randint(6, 60, n),
        "sex": np.random.choice(["Male", "Female"], n),
        "weight_kg": np.round(np.random.normal(12, 2.5, n), 1),
        "height_cm": np.round(np.random.normal(90, 8, n), 1),
        "wfh_zscore": np.round(np.random.normal(-0.5, 1.2, n), 2),
    })
    df["malnutrition_type"] = pd.cut(
        df["wfh_zscore"],
        bins=[-10, -3, -2, 100],
        labels=["Severe wasting", "Moderate wasting", "Normal"],
    )
    return df


def create_dummy_consumption_production(n=500):
    np.random.seed(45)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "monthly_food_expense": np.round(np.random.normal(120, 40, n), 2),
        "monthly_income": np.round(np.random.normal(300, 100, n), 2),
        "produces_own_food": np.random.choice([0, 1], n),
        "livestock_count": np.random.poisson(3, n),
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


def create_dummy_youth_decision(n=500):
    np.random.seed(46)
    df = pd.DataFrame({
        "youth_id": range(1, n + 1),
        "age": np.random.randint(15, 29, n),
        "sex": np.random.choice(["Male", "Female"], n),
        "education": np.random.choice(["None", "Primary", "Secondary", "Tertiary"], n),
        "employment_status": np.random.choice(
            ["Unemployed", "Employed", "Self-employed", "Student"], n
        ),
        "decision_power_score": np.random.randint(1, 6, n),
        "agency_score": np.random.randint(1, 6, n),
        "hope_future_score": np.random.randint(1, 6, n),
        "financial_literacy_score": np.random.randint(1, 6, n),
        "empathy_score": np.random.randint(1, 6, n),
        "participation_score": np.random.randint(1, 6, n),
    })
    df["received_training"] = np.random.choice([0, 1], n)
    return df


def create_dummy_integrated(n=500):
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
        df,
        question_cols=[f"hfias_q{i}" for i in range(1, 10)],
        score_col="hfias_score",
        category_col="hfias_category",
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
# SIDEBAR: CHOOSE SAMPLE / UPLOAD / KOBO
# ==================================================
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use sample (dummy) dataset", "Upload your own dataset", "Load from KoboToolbox"],
)

df = None
dataset_label = None

# ---------- 1) SAMPLE (DUMMY) DATA ----------
if data_source == "Use sample (dummy) dataset":
    sample_choice = st.sidebar.selectbox(
        "Select sample dataset",
        [
            "HFIAS household food insecurity",
            "Women’s dietary diversity (WDDS)",
            "Child malnutrition / anthropometry",
            "Food consumption & production",
            "Youth development & decision-making",
            "Integrated multi-topic dataset",
        ],
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

# ---------- 2) UPLOAD LOCAL FILE ----------
elif data_source == "Upload your own dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, Excel, JSON, TSV, Stata, SPSS, or PDF file",
        type=["csv", "xlsx", "xls", "json", "tsv", "txt", "dta", "sav", "pdf"],
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
                # PDF support: convert pages to text using pdfplumber
                try:
                    import pdfplumber

                    pages = []
                    with pdfplumber.open(uploaded_file) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text() or ""
                            pages.append({"page": i + 1, "text": text})
                    df = pd.DataFrame(pages)
                    dataset_label = "PDF text (page-level) dataset"
                    st.info(
                        "PDF loaded as page-level text. "
                        "You can generate AI narrative from this text, but numeric analysis may be limited."
                    )
                except Exception as e:
                    st.error(
                        f"Error reading PDF with pdfplumber: {e}. "
                        "Check that 'pdfplumber' is in your requirements.txt."
                    )
                    df = None
            else:
                st.error("Unsupported file format.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# ---------- 3) LOAD FROM KOBOTOOLBOX ----------
elif data_source == "Load from KoboToolbox":
    st.sidebar.markdown("### KoboToolbox Connection")

    # Default to EU server + your known asset UID
    kobo_server = st.sidebar.text_input(
        "Kobo server URL",
        value="https://eu.kobotoolbox.org",
        help="Example: https://kf.kobotoolbox.org or https://eu.kobotoolbox.org",
    )

    default_kobo_token = st.secrets.get("KOBO_TOKEN", "")
    kobo_token = st.sidebar.text_input(
        "Kobo API Token",
        type="password",
        value=default_kobo_token,
        help="Paste your Kobo API token here (Profile → API token). "
             "For deployment, set KOBO_TOKEN in Streamlit secrets instead of hardcoding.",
    )

    kobo_asset_uid = st.sidebar.text_input(
        "Form / Asset UID",
        value="aRcHyjoYEn6fCkHuEX4QYm",
        help="The UID of your Kobo form (e.g., aRcHyjoYEn6fCkHuEX4QYm).",
    )

    load_kobo = st.sidebar.button("Load data from Kobo")

    if load_kobo:
        if not (kobo_server and kobo_token and kobo_asset_uid):
            st.error("Please provide server URL, API token, and asset UID.")
        else:
            try:
                kobo_server = kobo_server.rstrip("/")
                url = f"{kobo_server}/api/v2/assets/{kobo_asset_uid}/data/?format=json"

                headers = {
                    "Authorization": f"Token {kobo_token}",
                }

                st.info("Requesting data from KoboToolbox...")
                resp = requests.get(url, headers=headers)

                if resp.status_code != 200:
                    st.error(
                        f"Error fetching data from Kobo (status {resp.status_code}): "
                        f"{resp.text[:400]}"
                    )
                else:
                    data_json = resp.json()

                    if "results" in data_json:
                        records = data_json["results"]
                    else:
                        records = data_json

                    if not records:
                        st.warning("No submissions found for this asset.")
                    else:
                        df = pd.DataFrame.from_records(records)
                        dataset_label = f"Kobo data (asset {kobo_asset_uid})"
                        st.success("Data loaded successfully from KoboToolbox.")

                        # Optional: drop Kobo system columns (starting with "_")
                        system_cols = [c for c in df.columns if c.startswith("_")]
                        if system_cols:
                            st.info(
                                f"Dropping Kobo system columns from analysis: {system_cols}"
                            )
                            df = df.drop(columns=system_cols)

            except Exception as e:
                st.error(f"Error loading data from KoboToolbox: {e}")

# ---------- Final check ----------
if df is None:
    st.info("Select a sample dataset, upload a file, or load from KoboToolbox to begin.")
    st.stop()

# Show basic info
st.success(f"Dataset loaded: {dataset_label or 'Uploaded dataset'}")
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

# Quick overview, especially useful for Kobo data
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if dataset_label and "Kobo" in dataset_label:
    st.markdown("### Kobo Dataset Variable Overview")
    st.write(f"Numeric columns ({len(numeric_cols)}):", numeric_cols)
    st.write(f"Categorical/text columns ({len(categorical_cols)}):", categorical_cols)

# ==================================================
# ANALYSIS MODE: AI AUTO vs SCRIPT
# ==================================================
st.sidebar.header("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose analysis mode",
    ["AI auto-detect analysis", "Use custom analysis script"],
)

narrative_chunks = []  # for AI narrative later

# ==================================================
# SIDEBAR CONTROLS FOR CATEGORICAL DESCRIPTIVES & CROSSTABS
# ==================================================
st.sidebar.header("Descriptive & Crosstab Options")

selected_numeric_by_group = st.sidebar.multiselect(
    "Numeric variables for descriptive by categories (mean, std, etc.)",
    options=numeric_cols,
)

selected_group_vars = st.sidebar.multiselect(
    "Grouping categorical variables (e.g., region, sex, education)",
    options=categorical_cols,
)

crosstab_var1 = st.sidebar.selectbox(
    "Crosstab variable 1 (categorical)",
    ["(none)"] + categorical_cols if categorical_cols else ["(none)"],
)

crosstab_var2 = st.sidebar.selectbox(
    "Crosstab variable 2 (categorical)",
    ["(none)"] + categorical_cols if categorical_cols else ["(none)"],
)

selected_cats_for_freq = st.sidebar.multiselect(
    "Categorical variables for frequency plots (bar/pie)",
    options=categorical_cols,
)

# ==================================================
# CUSTOM SCRIPT MODE (STUB + BASIC ANALYSIS)
# ==================================================
if analysis_mode == "Use custom analysis script":
    st.subheader("Custom Script–Driven Analysis")

    script_type = st.selectbox(
        "Type of script you want to reference",
        ["Stata (.do)", "SPSS (.sps)", "Python code (text)"],
    )

    script_text = st.text_area(
        "Paste your analysis script here (for now, used as guidance for AI + summaries)",
        height=200,
        placeholder="Paste your .do / .sps / Python logic here...",
    )

    st.markdown("### Basic descriptive analysis based on your dataset")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)
        narrative_chunks.append(
            "Descriptive statistics (custom script mode):\n" + desc.to_string()
        )
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

    # ------------------ HFIAS ------------------
    if "hfias_score" in df.columns:
        st.markdown("### HFIAS Summary")
        hfias = df["hfias_score"]
        st.write("HFIAS score descriptive statistics:")
        st.dataframe(hfias.describe().to_frame().T)

        cat_col = "hfias_category" if "hfias_category" in df.columns else None
        if not cat_col:
            df = compute_hfias_scores(
                df,
                question_cols=None,
                score_col="hfias_score",
                category_col="hfias_category",
            )
            cat_col = "hfias_category"

        st.write(
            "HFIAS severity distribution (count, mean score, and % of total by category):"
        )
        total_n = len(df)
        hfias_summary = (
            df.groupby(cat_col)["hfias_score"].agg(count="size", mean="mean").reset_index()
        )
        hfias_summary["percent"] = (hfias_summary["count"] / total_n * 100).round(1)
        hfias_summary["mean"] = hfias_summary["mean"].round(2)
        st.dataframe(hfias_summary)

        st.bar_chart(hfias_summary.set_index(cat_col)["count"])

        narrative_chunks.append(
            "HFIAS distribution (count, mean score, and % of total by category):\n"
            + hfias_summary.to_string(index=False)
        )

    # ------------------ HDDS / WDDS ------------------
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
                f"{col.upper()} distribution summary:\n"
                + df[col].describe().to_string()
            )

    # ------------------ CHILD ANTHROPOMETRY ------------------
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
            mal_tbl = pd.DataFrame({
                "count": mal_counts,
                "percent": (mal_counts / mal_counts.sum() * 100).round(1),
            })
            st.dataframe(mal_tbl)
            st.bar_chart(mal_counts)
            narrative_chunks.append(
                "Malnutrition type frequencies (count and %):\n" + mal_tbl.to_string()
            )

    # ------------------ YOUTH DEVELOPMENT & EMPOWERMENT ------------------
    st.markdown("## Youth Development & Empowerment Analysis")

    # Keywords for youth numeric indicators
    youth_num_keywords = [
        "decision", "power", "agency", "aspiration", "hope", "future",
        "financial", "finance", "literacy", "saving", "savings",
        "budget", "empathy", "empath", "participation", "voice", "leadership"
    ]

    # Keywords for youth categorical indicators
    youth_cat_keywords = [
        "youth", "training", "program", "cohort", "group",
        "employment", "employed", "self_emp", "business",
        "mentor", "club", "volunteer"
    ]

    youth_numeric_cols = [
        c for c in numeric_cols
        if any(k in c.lower() for k in youth_num_keywords)
    ]

    youth_categorical_cols = [
        c for c in categorical_cols
        if any(k in c.lower() for k in youth_num_keywords + youth_cat_keywords)
    ]

    if youth_numeric_cols or youth_categorical_cols:
        st.success(
            f"Detected youth-relevant variables: "
            f"{len(youth_numeric_cols)} numeric, {len(youth_categorical_cols)} categorical."
        )

        # ----- Numeric youth indicators -----
        if youth_numeric_cols:
            st.markdown("### Youth numeric indicators (decision power, hope, financial literacy, empathy, etc.)")
            youth_desc = df[youth_numeric_cols].describe().T
            st.dataframe(youth_desc)
            narrative_chunks.append(
                "Youth numeric indicators (decision/hope/financial/empathy) descriptives:\n"
                + youth_desc.to_string()
            )

            # Histograms for up to 8 youth numeric vars
            for col in youth_numeric_cols[:8]:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=10)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        # ----- Youth indicators by sex -----
        if youth_numeric_cols and "sex" in df.columns:
            st.markdown("#### Youth indicators by sex")
            for col in youth_numeric_cols:
                grouped = df.groupby("sex")[col].mean()
                st.write(f"Mean {col} by sex:")
                st.dataframe(grouped.to_frame())
                narrative_chunks.append(
                    f"Mean {col} by sex:\n{grouped.to_string()}"
                )

        # ----- Youth indicators by region -----
        if youth_numeric_cols and "region" in df.columns:
            st.markdown("#### Youth indicators by region")
            for col in youth_numeric_cols:
                grouped_r = df.groupby("region")[col].mean()
                st.write(f"Mean {col} by region:")
                st.dataframe(grouped_r.to_frame())
                narrative_chunks.append(
                    f"Mean {col} by region:\n{grouped_r.to_string()}"
                )

        # ----- Categorical youth variables -----
        if youth_categorical_cols:
            st.markdown("### Youth categorical variables (programs, groups, status, etc.)")
            total_n = len(df)
            for col in youth_categorical_cols:
                vc = df[col].value_counts(dropna=False)
                freq_tbl = pd.DataFrame({
                    "count": vc,
                    "percent": (vc / total_n * 100).round(1),
                })
                st.write(f"**{col}** (count and % of total):")
                st.dataframe(freq_tbl)
                narrative_chunks.append(
                    f"Youth categorical {col} (count and %):\n"
                    + freq_tbl.to_string()
                )

                # Bar chart
                fig, ax = plt.subplots()
                vc.plot(kind="bar", ax=ax)
                ax.set_title(f"{col} (bar chart)")
                ax.set_ylabel("Count")
                st.pyplot(fig)

                # Pie chart if categories not too many
                if vc.shape[0] <= 10:
                    fig2, ax2 = plt.subplots()
                    ax2.pie(
                        vc.values,
                        labels=vc.index.astype(str),
                        autopct="%1.1f%%",
                    )
                    ax2.set_title(f"{col} (pie chart)")
                    st.pyplot(fig2)
    else:
        st.info(
            "No youth-specific variables detected by name. "
            "You can still use the general descriptive and crosstab options in the sidebar."
        )

    # ==================================================
    # DESCRIPTIVE ANALYSIS BY CATEGORICAL VARIABLES + CROSSTABS
    # ==================================================
    st.markdown("### Descriptive Analysis by Categorical Variables")

    left_col, right_col = st.columns([2, 1])

    # ----- Numeric by group (mean, std, count, etc.) -----
    if selected_numeric_by_group and selected_group_vars:
        with left_col:
            st.markdown("#### Numeric variables by selected categories")

            total_n = len(df)
            group_counts = df.groupby(selected_group_vars).size().rename("group_count")
            group_perc = (group_counts / total_n * 100).rename("group_percent")

            grouped_stats = df.groupby(selected_group_vars)[selected_numeric_by_group].agg(
                ["mean", "std", "count", "min", "max"]
            )

            grouped = pd.concat([grouped_stats, group_counts, group_perc], axis=1)

            new_cols = []
            for col in grouped.columns:
                if isinstance(col, tuple):
                    new_cols.append(
                        "_".join([str(c) for c in col if c != ""]).strip("_")
                    )
                else:
                    new_cols.append(str(col))
            grouped.columns = new_cols

            st.dataframe(grouped)

            narrative_chunks.append(
                f"Numeric descriptives by {selected_group_vars} for {selected_numeric_by_group} "
                f"(including group_count and group_percent):\n{grouped.to_string()}"
            )

        primary_group = selected_group_vars[0]
        with right_col:
            st.markdown(f"#### Mean by {primary_group}")
            for num_var in selected_numeric_by_group:
                fig, ax = plt.subplots()
                df.groupby(primary_group)[num_var].mean().plot(kind="bar", ax=ax)
                ax.set_title(f"Mean {num_var} by {primary_group}")
                ax.set_ylabel(num_var)
                ax.set_xlabel(primary_group)
                st.pyplot(fig)

    # ----- Categorical frequency plots (bar / pie) -----
    if selected_cats_for_freq:
        with left_col:
            st.markdown("#### Frequency tables for selected categoricals")
            total_n = len(df)
            for cat in selected_cats_for_freq:
                vc = df[cat].value_counts(dropna=False)
                freq_tbl = pd.DataFrame({
                    "count": vc,
                    "percent": (vc / total_n * 100).round(1),
                })
                st.write(f"**{cat}**")
                st.dataframe(freq_tbl)
                narrative_chunks.append(
                    f"Frequencies for {cat} (count and % of total):\n"
                    + freq_tbl.to_string()
                )

        with right_col:
            st.markdown("#### Categorical plots")
            for cat in selected_cats_for_freq:
                vc = df[cat].value_counts(dropna=False)

                fig, ax = plt.subplots()
                vc.plot(kind="bar", ax=ax)
                ax.set_title(f"{cat} (bar chart)")
                ax.set_ylabel("Count")
                st.pyplot(fig)

                if vc.shape[0] <= 10:
                    fig2, ax2 = plt.subplots()
                    ax2.pie(
                        vc.values,
                        labels=vc.index.astype(str),
                        autopct="%1.1f%%",
                    )
                    ax2.set_title(f"{cat} (pie chart)")
                    st.pyplot(fig2)

    # ----- Crosstab between two categorical variables -----
    if (
        crosstab_var1 != "(none)"
        and crosstab_var2 != "(none)"
        and crosstab_var1 != crosstab_var2
    ):
        st.markdown("### Crosstab between two categorical variables")
        with left_col:
            xtab = pd.crosstab(
                df[crosstab_var1], df[crosstab_var2], dropna=False
            )
            st.write(f"**Crosstab: {crosstab_var1} × {crosstab_var2} (counts)**")
            st.dataframe(xtab)
            narrative_chunks.append(
                f"Crosstab counts for {crosstab_var1} x {crosstab_var2}:\n"
                + xtab.to_string()
            )

            xtab_pct = xtab.div(xtab.sum(axis=1), axis=0) * 100
            st.write(f"**Crosstab: {crosstab_var1} × {crosstab_var2} (row %)**")
            st.dataframe(xtab_pct.round(1))

        with right_col:
            fig, ax = plt.subplots()
            xtab_pct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"{crosstab_var1} × {crosstab_var2} (row % stacked)")
            ax.set_ylabel("Percentage")
            ax.set_xlabel(crosstab_var1)
            st.pyplot(fig)

    # ==================================================
    # GENERAL DESCRIPTIVES
    # ==================================================
    st.markdown("### General Descriptive Statistics")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)
        narrative_chunks.append("Overall numeric descriptives:\n" + desc.to_string())
    if categorical_cols:
        st.markdown("### Key Categorical Distributions (all)")
        total_n = len(df)
        for col in categorical_cols[:10]:
            st.write(f"**{col}** value counts and % of total:")
            vc = df[col].value_counts(dropna=False)
            freq_tbl = pd.DataFrame({
                "count": vc,
                "percent": (vc / total_n * 100).round(1),
            })
            st.dataframe(freq_tbl)
            narrative_chunks.append(
                f"Value counts for {col} (count and %):\n"
                + freq_tbl.to_string()
            )


# ==================================================
# AI NARRATIVE REPORT
# ==================================================
st.markdown("---")
st.subheader("AI Narrative Report")

if not openai.api_key:
    st.warning(
        "OpenAI API key not set in Streamlit secrets. Narrative report not available."
    )
else:
    user_prompt = st.text_area(
        "Optional: refine what you want the AI to focus on",
        value=(
            "Summarize the dataset, describe key findings on food security, "
            "dietary diversity, youth decision-making and agency, hope for the future, "
            "financial literacy, empathy, and income/consumption where applicable. "
            "Highlight any gender or regional differences and potential program implications."
        ),
    )

    if st.button("Generate AI Narrative"):
        context_sample = df.head(10).to_dict()
        combined_narrative = (
            "\n\n".join(narrative_chunks) if narrative_chunks else "No prior summaries."
        )

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
    mime="text/csv",
)

pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer) as pdf:
    for col in numeric_cols[:12]:
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
    mime="application/pdf",
)
