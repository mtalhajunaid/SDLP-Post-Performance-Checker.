
import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD SAVED MODEL
# ============================================================

final_clf = joblib.load("final_top25_classifier.pkl")
feature_cols = joblib.load("final_feature_columns.pkl")
topic_flag_cols = joblib.load("topic_flag_columns.pkl")


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
# BUILD INPUT ROW
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

    # Numeric variables
    if "Post Hour" in row.columns:
        row.loc[0, "Post Hour"] = float(hour)

    if "Posting Frequency (posts/week)" in row.columns:
        row.loc[0, "Posting Frequency (posts/week)"] = float(posting_frequency)

    if "Current Followers" in row.columns:
        row.loc[0, "Current Followers"] = float(followers)

    # Account
    account_col = f"Account_{account}"
    if account_col in row.columns:
        row.loc[0, account_col] = 1

    # Platform
    platform_col = f"Platform_{platform}"
    if platform_col in row.columns:
        row.loc[0, platform_col] = 1

    # Format
    format_col = f"Post Format (Clean)_{post_format}"
    if format_col in row.columns:
        row.loc[0, format_col] = 1

    # Day
    day_col = f"Day of Week_{day}"
    if day_col in row.columns:
        row.loc[0, day_col] = 1

    # Topics
    selected_topics = selected_topics or []

    for topic in selected_topics:
        if topic in row.columns:
            row.loc[0, topic] = 1

    return row


# ============================================================
# PREDICTION
# ============================================================

def analyse_post(
    account,
    platform,
    post_format,
    day,
    hour,
    posting_frequency,
    followers,
    selected_topics
):

    if followers is None or followers <= 0:
        return (
            "### Please enter the current follower count.",
            ""
        )

    row = build_model_row(
        account,
        platform,
        post_format,
        day,
        hour,
        posting_frequency,
        followers,
        selected_topics
    )

    probability = float(
        final_clf.predict_proba(row)[0, 1]
    )

    predicted_class = int(
        final_clf.predict(row)[0]
    )

    pct = probability * 100


    if probability < 0.35:
        level = "Lower historical likelihood"

    elif probability < 0.50:
        level = "Moderate historical likelihood"

    elif probability < 0.65:
        level = "Higher historical likelihood"

    else:
        level = "Strong historical signal"


    if predicted_class == 1:
        flag = "The model flags this as a potential Top-25% post."
    else:
        flag = "The model does not currently flag this as a Top-25% post."


    result = f"""
# {pct:.1f}%

### Estimated Top-25% probability

**{level}**

{flag}
"""


    topics_text = (
        ", ".join(selected_topics)
        if selected_topics
        else "None selected"
    )

    details = f"""
### Inputs used

**Account:** {account}  
**Platform:** {platform}  
**Format:** {post_format}  
**Day:** {day}  
**Posting time:** {int(hour):02d}:00  
**Posting frequency:** {posting_frequency:.1f} posts/week  
**Current followers:** {int(followers):,}  
**Topics:** {topics_text}

---

This tool uses historical SDLP social-media performance patterns to support
pre-post decision making. The output should be treated as a decision-support
signal rather than a guarantee of future engagement.
"""

    return result, details


# ============================================================
# SDLP DESIGN
# ============================================================

css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: Arial, Helvetica, sans-serif;
}

#sdlp-header {
    background: linear-gradient(135deg, #006747, #004c36);
    color: white;
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 20px;
}

#sdlp-header h1,
#sdlp-header p {
    color: white !important;
}

#result-card {
    padding: 15px;
    border-radius: 18px;
}

footer {
    visibility: hidden;
}
"""


with gr.Blocks(
    title="SDLP Post Performance Checker",
    css=css
) as demo:

    gr.Markdown(
        """
# SDLP Post Performance Checker

### Data-driven pre-post decision support

Enter the details of a proposed Facebook or Instagram post to estimate
whether its characteristics are similar to historically high-performing
SDLP posts.

**For the most accurate result, enter the account's current follower count.**
        """,
        elem_id="sdlp-header"
    )


    with gr.Row():

        with gr.Column():

            gr.Markdown("### Post details")

            account = gr.Dropdown(
                accounts,
                value="Social Democratic and Labour Party",
                label="Account"
            )

            platform = gr.Radio(
                platforms,
                value="Facebook",
                label="Platform"
            )

            followers = gr.Number(
                label="Current follower count",
                precision=0,
                info="Enter the live follower count for the selected account and platform."
            )

            post_format = gr.Dropdown(
                formats,
                value="Image",
                label="Post format"
            )


        with gr.Column():

            gr.Markdown("### Timing and content")

            day = gr.Dropdown(
                days,
                value="Monday",
                label="Day of week"
            )

            hour = gr.Slider(
                minimum=0,
                maximum=23,
                value=12,
                step=1,
                label="Planned posting hour"
            )

            posting_frequency = gr.Slider(
                minimum=0,
                maximum=30,
                value=7,
                step=0.5,
                label="Posting frequency (posts/week)"
            )

            topics = gr.CheckboxGroup(
                choices=topic_flag_cols,
                label="Political topic(s)",
                info="Select all that apply."
            )


    analyse = gr.Button(
        "Analyse Post",
        variant="primary"
    )


    gr.Markdown("---")

    with gr.Row():

        with gr.Column():
            result = gr.Markdown(
                elem_id="result-card"
            )

        with gr.Column():
            details = gr.Markdown()


    analyse.click(
        fn=analyse_post,
        inputs=[
            account,
            platform,
            post_format,
            day,
            hour,
            posting_frequency,
            followers,
            topics
        ],
        outputs=[
            result,
            details
        ]
    )


    gr.Markdown(
        """
---
**About this tool:** Developed as part of an MSc Strategic Business Analytics
research project using historical SDLP Facebook and Instagram data.
"""
    )


if __name__ == "__main__":
    demo.launch()
