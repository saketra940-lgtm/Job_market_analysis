import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="JobMatch AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    .job-card {
        border: 1px solid #dfe5ec;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 12px 0 4px 0;
        background: white;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .note-box {
        padding: 14px 16px;
        border-left: 4px solid #2563eb;
        background: #eff6ff;
        border-radius: 8px;
        color: #1e3a5f;
        margin-bottom: 20px;
    }

    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "all_upwork_jobs_2024-02-07-2024-03-24.csv"


# =========================================================
# DATA FUNCTIONS
# =========================================================

def prepare_market_data(dataframe):
    data = dataframe.copy()

    required_columns = [
        "title",
        "published_date",
        "is_hourly",
        "hourly_low",
        "hourly_high",
        "budget",
        "country"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    data["published_date"] = pd.to_datetime(
        data["published_date"],
        errors="coerce"
    )

    data["month"] = (
        data["published_date"]
        .dt.to_period("M")
        .astype(str)
    )

    data["title"] = data["title"].fillna("Unknown Job")
    data["country"] = data["country"].fillna("Unknown")

    return data


@st.cache_data
def load_market_data():
    data = pd.read_csv(CSV_FILE)
    return prepare_market_data(data)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# =========================================================
# HEADER
# =========================================================

st.title("💼 JobMatch AI")

st.subheader(
    "Job Market Analytics & Personalized Recommendation System"
)

st.write(
    "Use the recommendation engine to find relevant jobs, "
    "or switch to the market dashboard to analyze monthly trends."
)

st.divider()


# =========================================================
# LOAD CSV FOR TASK 6
# =========================================================

try:
    market_df = load_market_data()

except FileNotFoundError:
    market_df = None

except Exception as e:
    st.error(f"Could not load market dataset: {e}")
    st.stop()


# =========================================================
# NAVIGATION
# =========================================================

with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Select section",
        [
            "🔎 Job Recommendations",
            "📊 Market Dashboard"
        ]
    )

    st.divider()


# =========================================================
# TASK 5 — RECOMMENDATION SYSTEM
# =========================================================

if page == "🔎 Job Recommendations":

    with st.sidebar:
        st.header("Job Preferences")

        user_query = st.text_input(
            "Skills / Desired Job Role",
            placeholder="e.g. Python Data Analyst"
        )

        country = st.text_input(
            "Country",
            value="All",
            placeholder="e.g. India"
        )

        job_type = st.selectbox(
            "Job Type",
            ["All", "Hourly", "Fixed Price"]
        )

        top_n = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=10
        )

        recommend_button = st.button(
            "🔎 Find Matching Jobs",
            use_container_width=True
        )

        st.divider()

        try:
            health = requests.get(
                "http://127.0.0.1:5000/",
                timeout=3
            )

            if health.status_code == 200:
                st.success("🟢 Recommendation API Online")
            else:
                st.warning("🟡 API Response Issue")

        except requests.exceptions.RequestException:
            st.error("🔴 Recommendation API Offline")

    st.header("💼 Personalized Job Recommendation System")

    st.markdown(
        """
        <div class="note-box">
        Enter your skills or desired job role. The existing Flask API
        uses TF-IDF and Cosine Similarity to rank relevant jobs from
        the provided job-posting dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    if recommend_button:

        if not user_query.strip():
            st.warning(
                "Please enter your skills or desired job role."
            )

        else:

            payload = {
                "user_query": user_query.strip(),
                "country": country,
                "job_type": job_type,
                "top_n": top_n
            }

            try:
                with st.spinner(
                    "Finding the best matching jobs..."
                ):
                    response = requests.post(
                        "http://127.0.0.1:5000/recommend",
                        json=payload,
                        timeout=120
                    )

                if response.status_code != 200:
                    st.error(
                        f"API returned status code "
                        f"{response.status_code}"
                    )
                    try:
                        st.json(response.json())
                    except Exception:
                        st.code(response.text)

                else:

                    result = response.json()
                    recommendations = result.get(
                        "recommendations",
                        []
                    )

                    if not recommendations:

                        st.warning(
                            "No matching jobs were found. "
                            "Try a different role, country, or job type."
                        )

                    else:

                        st.subheader(
                            f'Recommended Jobs for "{user_query}"'
                        )

                        hourly_jobs = sum(
                            1
                            for job in recommendations
                            if job.get("is_hourly")
                        )

                        fixed_jobs = (
                            len(recommendations)
                            - hourly_jobs
                        )

                        scores = [
                            safe_float(
                                job.get("similarity_score", 0)
                            )
                            for job in recommendations
                        ]

                        average_match = (
                            float(np.mean(scores))
                            if scores
                            else 0
                        )

                        m1, m2, m3, m4 = st.columns(4)

                        m1.metric(
                            "Jobs Found",
                            len(recommendations)
                        )

                        m2.metric(
                            "Hourly Jobs",
                            hourly_jobs
                        )

                        m3.metric(
                            "Fixed Price",
                            fixed_jobs
                        )

                        m4.metric(
                            "Average Match",
                            f"{average_match * 100:.1f}%"
                        )

                        st.info(
                            "Primary job information comes from the "
                            "provided dataset. View Job opens the "
                            "original external link."
                        )

                        for i, job in enumerate(
                            recommendations,
                            start=1
                        ):

                            title = str(
                                job.get(
                                    "title",
                                    "Untitled Job"
                                )
                            )

                            country_value = str(
                                job.get(
                                    "country",
                                    "Not specified"
                                )
                            )

                            score = (
                                safe_float(
                                    job.get(
                                        "similarity_score",
                                        0
                                    )
                                )
                                * 100
                            )

                            if job.get("is_hourly"):

                                low = job.get("hourly_low")
                                high = job.get("hourly_high")

                                if (
                                    pd.notna(low)
                                    and pd.notna(high)
                                ):
                                    compensation = (
                                        f"${safe_float(low):.2f} - "
                                        f"${safe_float(high):.2f} / hr"
                                    )
                                else:
                                    compensation = (
                                        "Rate not available"
                                    )

                                type_text = "Hourly"

                            else:

                                budget = job.get("budget")

                                if pd.notna(budget):
                                    compensation = (
                                        f"${safe_float(budget):,.2f} fixed"
                                    )
                                else:
                                    compensation = (
                                        "Budget not available"
                                    )

                                type_text = "Fixed Price"

                            with st.container(border=True):

                                st.caption(
                                    f"RECOMMENDATION #{i}"
                                )

                                st.markdown(
                                    f"### {title}"
                                )

                                st.write(
                                    f"📍 **{country_value}**   •   "
                                    f"💼 **{type_text}**   •   "
                                    f"💰 **{compensation}**"
                                )

                                d1, d2, d3, d4 = st.columns(
                                    [2, 2, 2, 1]
                                )

                                with d1:
                                    st.caption("COUNTRY")
                                    st.write(country_value)

                                with d2:
                                    st.caption("COMPENSATION")
                                    st.write(compensation)

                                with d3:
                                    st.caption("JOB TYPE")
                                    st.write(type_text)

                                with d4:
                                    st.metric(
                                        "MATCH",
                                        f"{score:.1f}%"
                                    )

                                link = job.get("link")

                                if link:
                                    st.link_button(
                                        "↗ View Job",
                                        str(link)
                                    )

                                st.divider()

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the Flask API."
                )

                st.info(
                    "Run 'python api.py' in another "
                    "PowerShell window and keep it running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The recommendation request timed out."
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"API request failed: {e}"
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )

    else:

        st.subheader("How It Works")

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "01",
                "Enter Skills",
                "Desired role or keywords"
            )

        with b:
            st.metric(
                "02",
                "Set Filters",
                "Country and job type"
            )

        with c:
            st.metric(
                "03",
                "Get Matches",
                "Ranked recommendations"
            )


# =========================================================
# TASK 6 — MONTHLY MARKET DASHBOARD
# =========================================================

else:

    st.header(
        "📊 Monthly Job Market Dynamics Dashboard"
    )

    st.markdown(
        """
        <div class="note-box">
        This dashboard tracks changes in job-market dynamics
        using the provided job-posting dataset. If a new monthly
        CSV is supplied, use the refresh button to recalculate trends.
        </div>
        """,
        unsafe_allow_html=True
    )

    # CSV available?
    if market_df is None:

        st.error(
            f"'{CSV_FILE}' was not found in the same folder as app.py."
        )

        st.info(
            "Put the original CSV in the same folder as app.py "
            "and refresh the page."
        )

        uploaded_file = st.file_uploader(
            "Or upload the original CSV here",
            type=["csv"]
        )

        if uploaded_file is None:
            st.stop()

        try:
            uploaded_df = pd.read_csv(uploaded_file)
            market_df = prepare_market_data(uploaded_df)

        except Exception as e:
            st.error(
                f"Could not read the uploaded CSV: {e}"
            )
            st.stop()

    if st.button(
        "🔄 Refresh Dashboard Data"
    ):

        load_market_data.clear()
        st.rerun()

    # Month filter
    available_months = sorted(
        market_df["month"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_months = st.multiselect(
        "Select Months",
        options=available_months,
        default=available_months
    )

    if selected_months:
        dashboard_df = market_df[
            market_df["month"].isin(selected_months)
        ].copy()
    else:
        dashboard_df = market_df.copy()

    # KPI cards
    total_jobs = len(dashboard_df)

    hourly_jobs = int(
        (
            dashboard_df["is_hourly"] == True
        ).sum()
    )

    fixed_jobs = int(
        (
            dashboard_df["is_hourly"] == False
        ).sum()
    )

    countries = int(
        dashboard_df["country"].nunique()
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Total Job Postings",
        f"{total_jobs:,}"
    )

    k2.metric(
        "Hourly Jobs",
        f"{hourly_jobs:,}"
    )

    k3.metric(
        "Fixed Price Jobs",
        f"{fixed_jobs:,}"
    )

    k4.metric(
        "Countries",
        f"{countries:,}"
    )

    st.divider()

    # Chart 1
    st.subheader(
        "1. Monthly Job Posting Trend"
    )

    monthly_jobs = (
        dashboard_df
        .groupby("month")
        .size()
        .reset_index(
            name="job_postings"
        )
    )

    fig1 = px.line(
        monthly_jobs,
        x="month",
        y="job_postings",
        markers=True,
        title="Total Job Postings by Month"
    )

    fig1.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Job Postings",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # Chart 2
    st.subheader(
        "2. Monthly Hourly vs Fixed Price Jobs"
    )

    monthly_type = (
        dashboard_df
        .groupby(
            ["month", "is_hourly"]
        )
        .size()
        .reset_index(
            name="job_count"
        )
    )

    monthly_type["job_type"] = (
        monthly_type["is_hourly"].map(
            {
                True: "Hourly",
                False: "Fixed Price"
            }
        )
    )

    fig2 = px.bar(
        monthly_type,
        x="month",
        y="job_count",
        color="job_type",
        barmode="group",
        title="Hourly vs Fixed Price by Month"
    )

    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Jobs"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Chart 3
    st.subheader(
        "3. Average Hourly Rate Trend"
    )

    hourly_df = dashboard_df[
        dashboard_df["is_hourly"] == True
    ].copy()

    hourly_df["avg_hourly_rate"] = (
        hourly_df["hourly_low"]
        + hourly_df["hourly_high"]
    ) / 2

    monthly_rate = (
        hourly_df
        .dropna(
            subset=["avg_hourly_rate"]
        )
        .groupby("month")[
            "avg_hourly_rate"
        ]
        .mean()
        .reset_index()
    )

    if not monthly_rate.empty:

        fig3 = px.line(
            monthly_rate,
            x="month",
            y="avg_hourly_rate",
            markers=True,
            title="Average Hourly Rate by Month"
        )

        fig3.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Hourly Rate ($)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    else:

        st.info(
            "No valid hourly-rate data is available."
        )

    # Chart 4
    st.subheader(
        "4. Top Job Titles"
    )

    top_titles = (
        dashboard_df["title"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_titles.columns = [
        "title",
        "job_count"
    ]

    fig4 = px.bar(
        top_titles.sort_values("job_count"),
        x="job_count",
        y="title",
        orientation="h",
        title="Top 10 Job Titles"
    )

    fig4.update_layout(
        xaxis_title="Number of Postings",
        yaxis_title="Job Title"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # Chart 5
    st.subheader(
        "5. Top Countries by Job Postings"
    )

    country_jobs = (
        dashboard_df
        .groupby("country")
        .size()
        .reset_index(
            name="job_count"
        )
        .sort_values(
            "job_count",
            ascending=False
        )
        .head(10)
    )

    fig5 = px.bar(
        country_jobs.sort_values("job_count"),
        x="job_count",
        y="country",
        orientation="h",
        title="Top 10 Countries"
    )

    fig5.update_layout(
        xaxis_title="Number of Postings",
        yaxis_title="Country"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # Monthly table
    st.subheader(
        "6. Monthly Job Market Summary"
    )

    monthly_summary = (
        dashboard_df
        .groupby("month")
        .agg(
            total_jobs=("title", "count"),
            hourly_jobs=("is_hourly", "sum"),
            countries=("country", "nunique")
        )
        .reset_index()
    )

    monthly_summary["fixed_price_jobs"] = (
        monthly_summary["total_jobs"]
        - monthly_summary["hourly_jobs"]
    )

    monthly_summary = monthly_summary[
        [
            "month",
            "total_jobs",
            "hourly_jobs",
            "fixed_price_jobs",
            "countries"
        ]
    ]

    st.dataframe(
        monthly_summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        <b>Job Market Analysis & Recommendation System</b><br>
        Task 5: Personalized Job Recommendation Engine
        &nbsp; | &nbsp;
        Task 6: Monthly Job Market Dynamics Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
