import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="SDLP Post Performance Checker",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f7;
    }

    .main-title {
        background: linear-gradient(135deg, #006747, #004c36);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-title h1 {
        color: white;
        margin-bottom: 5px;
    }

    .main-title p {
        color: white;
        font-size: 17px;
        margin-bottom: 0;
    }

    .result-card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border-left: 7px solid #006747;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
        margin-top: 15px;
    }

    .small-note {
        color: #666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("final_top25_classifier.pkl")
    feature_cols = joblib.load("final_feature_columns.pkl")
    topic_cols = joblib.load("topic_flag_columns.pkl")

    return model, feature_cols, topic_cols


final_clf, feature_cols, topic_flag_cols = load_model()


# ============================================================
# OPTIONS
# ============================================================

accounts = [
    "Cara Hunter",
    "Claire Hanna",
    "Colin McGrath",
    "Daniel McCrossan",
    "Justin McNulty",
    "Mark H Durkan",
    "Matthew O'Toole",
    "Patsy McGlone",
    "Sinéad McLaughlin",
    "Social Democratic and Labour Party"
]

platforms = [
    "Facebook",
    "Instagram"
]

formats = [
    "Image",
    "Link-share",
    "Reel",
    "Text-only / Other",
    "Video"
]

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <h1>SDLP Post Performance Checker</h1>
        <p>
            Data-driven pre-post decision support using historical SDLP social media data.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter the details of a proposed Facebook or Instagram post to estimate "
    "whether its characteristics are similar to historically high-performing SDLP posts."
)


# ============================================================
# INPUTS
# ============================================================

left, right = st.columns(2)


with left:

    st.subheader("Post details")

    account = st.selectbox(
        "Account",
        accounts
    )

    platform = st.radio(
        "Platform",
        platforms,
        horizontal=True
    )

    followers = st.number_input(
        "Current follower count",
        min_value=1,
        value=1000,
        step=100,
        help="Enter the current follower count for the selected account and platform."
    )

    post_format = st.selectbox(
        "Post format",
        formats
    )


with right:

    st.subheader("Timing and content")

    day = st.selectbox(
        "Day of week",
        days
    )

    hour = st.slider(
        "Planned posting hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )

    posting_frequency = st.slider(
        "Posting frequency (posts/week)",
        min_value=0.0,
        max_value=30.0,
        value=7.0,
        step=0.5
    )

    selected_topics = st.multiselect(
        "Political topic(s)",
        topic_flag_cols
    )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_row(
    account,
    platform,
    post_format,
    day,
    hour,
    posting_frequency,
    followers,
    selected_topics
):

    row = pd.DataFrame(
        np.zeros((1, len(feature_cols))),
        columns=feature_cols
    )


    # Numeric features

    if "Post Hour" in row.columns:
        row.loc[0, "Post Hour"] = float(hour)

    if "Posting Frequency (posts/week)" in row.columns:
        row.loc[
            0,
            "Posting Frequency (posts/week)"
        ] = float(posting_frequency)

    if "Current Followers" in row.columns:
        row.loc[
            0,
            "Current Followers"
        ] = float(followers)


    # Account dummy

    account_col = f"Account_{account}"

    if account_col in row.columns:
        row.loc[0, account_col] = 1


    # Platform dummy

    platform_col = f"Platform_{platform}"

    if platform_col in row.columns:
        row.loc[0, platform_col] = 1


    # Format dummy

    format_col = f"Post Format (Clean)_{post_format}"

    if format_col in row.columns:
        row.loc[0, format_col] = 1


    # Day dummy

    day_col = f"Day of Week_{day}"

    if day_col in row.columns:
        row.loc[0, day_col] = 1


    # Topic flags

    selected_topics = selected_topics or []

    for topic in selected_topics:
        if topic in row.columns:
            row.loc[0, topic] = 1


    return row


# ============================================================
# ANALYSE BUTTON
# ============================================================

st.markdown("###")

analyse_clicked = st.button(
    "Analyse Post",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyse_clicked:

    row = build_model_row(
        account=account,
        platform=platform,
        post_format=post_format,
        day=day,
        hour=hour,
        posting_frequency=posting_frequency,
        followers=followers,
        selected_topics=selected_topics
    )


    probability = float(
        final_clf.predict_proba(row)[0, 1]
    )

    predicted_class = int(
        final_clf.predict(row)[0]
    )

    pct = probability * 100


    # Human-readable likelihood band

    if probability < 0.35:
        level = "Lower historical likelihood"

    elif probability < 0.50:
        level = "Moderate historical likelihood"

    elif probability < 0.65:
        level = "Higher historical likelihood"

    else:
        level = "Strong historical signal"


    if predicted_class == 1:

        flag = (
            "The model flags this as a potential Top-25% post."
        )

    else:

        flag = (
            "The model does not currently flag this as a Top-25% post."
        )


    # ========================================================
    # RESULT CARD
    # ========================================================

    st.markdown(
        f"""
        <div class="result-card">
            <h1>{pct:.1f}%</h1>
            <h3>Estimated Top-25% probability</h3>
            <p><strong>{level}</strong></p>
            <p>{flag}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader("Inputs used")

    summary_left, summary_right = st.columns(2)

    with summary_left:
        st.write(f"**Account:** {account}")
        st.write(f"**Platform:** {platform}")
        st.write(f"**Format:** {post_format}")
        st.write(f"**Day:** {day}")

    with summary_right:
        st.write(f"**Posting time:** {hour:02d}:00")
        st.write(
            f"**Posting frequency:** {posting_frequency:.1f} posts/week"
        )
        st.write(
            f"**Current followers:** {followers:,}"
        )

        if selected_topics:
            st.write(
                "**Topics:** " + ", ".join(selected_topics)
            )
        else:
            st.write("**Topics:** None selected")


    # ========================================================
    # INTERPRETATION NOTE
    # ========================================================

    st.info(
        "This result is based on historical SDLP social-media performance patterns. "
        "It should be used as a decision-support signal rather than a guarantee "
        "of future engagement."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Developed as part of an MSc Strategic Business Analytics research project."
)
