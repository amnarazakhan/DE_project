# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# import torch
# import sqlite3
#
# # Set up Streamlit page
# st.set_page_config(page_title="Social Media Bullying Trends", layout="wide")
#
# # === Load Model and Tokenizer ===
# @st.cache_resource
# def load_model():
#     model = RobertaForSequenceClassification.from_pretrained("amnarazakhan/roberta_cyberhate_trained")
#     tokenizer = RobertaTokenizer.from_pretrained("amnarazakhan/roberta_cyberhate_trained")
#     return model, tokenizer
#
# model, tokenizer = load_model()
#
# def is_bullying_model(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probs = torch.nn.functional.softmax(logits, dim=1)
#         predicted_class = torch.argmax(probs, dim=1).item()
#         confidence = probs[0][predicted_class].item()
#     return predicted_class, confidence
#
# # === Function to Connect to SQLite Database ===
# # def load_data():
# #     conn = sqlite3.connect('social_media.db')
# #     query = """
# #         SELECT id, title, text, url, score, comments, subreddit, timestamp_utc
# #         FROM reddit_raw
# #     """
# #     df = pd.read_sql_query(query, conn)
# #     conn.close()
# #
# #     if 'timestamp_utc' in df.columns:
# #         df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
# #     else:
# #         st.error("Column 'timestamp_utc' not found in database!")
# #
# #     return df
#
# def load_data():
#     import os
#     db_path = "db/social_media.db"
#   # adjust if path is different
#     conn = sqlite3.connect(db_path)
#
#     query = """
#         SELECT id, title, text, url, score, comments, subreddit, timestamp_utc
#         FROM reddit_raw
#     """
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#
#     if 'timestamp_utc' in df.columns:
#         df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
#     else:
#         st.error("Column 'timestamp_utc' not found in database!")
#
#     return df
#
#
# # === Load Dataset ===
# df = load_data()
# df['Platform'] = "Reddit"
# df.rename(columns={
#     'title': 'Topic',
#     'subreddit': 'Subreddit',
#     'score': 'Score',
#     'comments': 'Comments'
# }, inplace=True)
#
# # Add dummy label if not present (in case no labelling yet)
# if 'label' in df.columns:
#     df.rename(columns={'label': 'Bullying'}, inplace=True)
# else:
#     df['Bullying'] = 0
#
# df['Bullying'] = pd.to_numeric(df['Bullying'], errors='coerce').fillna(0).astype(int)
# df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
# df['Date'] = df['timestamp_utc'].dt.date
# df['Hour'] = df['timestamp_utc'].dt.hour
#
# st.title("📊 Social Media Bullying Trends Dashboard")
#
# # === Sidebar Filters ===
# st.sidebar.header("🔍 Filter Data")
# date_range = st.sidebar.date_input(
#     "Select Date Range",
#     [df['Date'].min(), df['Date'].max()]
# )
#
# platforms = st.sidebar.multiselect("Platforms", df['Platform'].unique(), default=df['Platform'].unique())
# subreddits = st.sidebar.multiselect("Subreddits", df['Subreddit'].unique(), default=df['Subreddit'].unique())
# bullying_only = st.sidebar.checkbox("Show only bullying posts")
#
# # === Apply Filters ===
# start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
# mask = (
#     (df['timestamp_utc'] >= start_date) &
#     (df['timestamp_utc'] <= end_date) &
#     (df['Platform'].isin(platforms)) &
#     (df['Subreddit'].isin(subreddits))
# )
# if bullying_only:
#     mask &= df['Bullying'] == 1
#
# filtered_df = df[mask]
#
# # === User Input for Classification ===
# st.markdown("### 📝 Enter a Post Title or Comment to Check for Bullying")
# user_input = st.text_area("Enter text here:")
#
# if user_input:
#     prediction, confidence = is_bullying_model(user_input)
#     if prediction == 1:
#         st.success(f"✅ This is likely **cyberbullying** (Confidence: {confidence:.2%})")
#     else:
#         st.info(f"🔍 This doesn't appear to be bullying (Confidence: {confidence:.2%})")
#
# # === KPIs ===
# st.markdown("### 📌 Key Metrics")
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Posts", len(filtered_df))
# col2.metric("% Bullying Posts", f"{(filtered_df['Bullying'].mean()) * 100:.1f}%")
# col3.metric("Most Active Subreddit", filtered_df['Subreddit'].mode().values[0] if not filtered_df.empty else "N/A")
# col4.metric("Top Platform", filtered_df['Platform'].mode().values[0] if not filtered_df.empty else "N/A")
#
# # === Trend Chart ===
# st.markdown("### 📈 Bullying Trend by Month and Year")
# filtered_df.loc[:, 'Year-Month'] = filtered_df['timestamp_utc'].dt.to_period('M').astype(str)
# trend_monthly = filtered_df[filtered_df['Bullying'] == 1].groupby('Year-Month').size().reset_index(name='Bullying Posts')
# fig_trend_monthly = px.line(trend_monthly, x='Year-Month', y='Bullying Posts',
#                             title="Bullying Posts per Month and Year",
#                             labels={'Year-Month': 'Month and Year', 'Bullying Posts': 'Number of Bullying Posts'})
# fig_trend_monthly.update_layout(xaxis_title="Month and Year", yaxis_title="Bullying Posts")
# st.plotly_chart(fig_trend_monthly, use_container_width=True)
#
# # === Top Subreddits Bar Chart ===
# st.markdown("### 📊 Top Subreddits by Bullying Posts")
# subreddit_stats = filtered_df[filtered_df['Bullying'] == 1]['Subreddit'].value_counts().reset_index()
# subreddit_stats.columns = ['Subreddit', 'Bullying Posts']
# fig_subreddit = px.bar(subreddit_stats, x='Subreddit', y='Bullying Posts', title="Top Subreddits with Bullying Posts")
# st.plotly_chart(fig_subreddit, use_container_width=True)
#
# # === Engagement: Score vs Comments ===
# st.markdown("### 🔥 Engagement by Score vs Comments")
# fig_engagement_bar = px.bar(
#     filtered_df,
#     x='Score',
#     y='Comments',
#     color=filtered_df['Bullying'].map({1: 'Bullying', 0: 'Non-Bullying'}),
#     hover_data=['Topic', 'Subreddit', 'url'],
#     title="Score vs Comments Engagement (Bar Chart)"
# )
# st.plotly_chart(fig_engagement_bar, use_container_width=True)
#
# # === Word Cloud ===
# st.markdown("### 🧠 Word Cloud of Topics")
# if not filtered_df.empty and filtered_df['Topic'].notna().any():
#     wordcloud_data = ' '.join(filtered_df['Topic'].dropna().astype(str))
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
#
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     st.pyplot(plt)
# else:
#     st.warning("No text data available for the word cloud with the current filters.")

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import sqlite3

# Set up Streamlit page
st.set_page_config(page_title="Social Media Bullying Trends", layout="wide")

# === Load Model and Tokenizer ===
@st.cache_resource
def load_model():
    model = RobertaForSequenceClassification.from_pretrained("amnarazakhan/roberta_cyberhate_trained")
    tokenizer = RobertaTokenizer.from_pretrained("amnarazakhan/roberta_cyberhate_trained")
    return model, tokenizer

model, tokenizer = load_model()

def is_bullying_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return predicted_class, confidence

# === Load and Join Data from SQLite ===
def load_data():
    db_path = "db/social_media.db"
    conn = sqlite3.connect(db_path)

    query = """
        SELECT 
            raw.id,
            raw.title AS Topic,
            raw.text,
            raw.url,
            raw.score AS Score,
            raw.comments AS Comments,
            raw.subreddit AS Subreddit,
            raw.timestamp_utc,
            lbl.label AS Bullying
        FROM reddit_raw raw
        LEFT JOIN reddit_cleaned cln ON raw.id = cln.id
        LEFT JOIN reddit_labelled lbl ON raw.id = lbl.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Handle timestamps
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
    else:
        st.error("Column 'timestamp_utc' not found in database!")

    return df

# === Load Dataset ===
df = load_data()
df['Platform'] = "Reddit"

# Prepare columns
df['Bullying'] = pd.to_numeric(df['Bullying'], errors='coerce').fillna(0).astype(int)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
df['Date'] = df['timestamp_utc'].dt.date
df['Hour'] = df['timestamp_utc'].dt.hour

# === Page Styling ===
st.markdown("""
    <style>
        .main { background-color: #white; color: 0e1117; }
        .st-bf, .st-ag, .st-af { background-color: #white; color: 0e1117; }
        .st-c4 { color: 0e1117; }
    </style>
""", unsafe_allow_html=True)

# === UI Title ===
st.title("📊 Social Media Cyberbullying Dashboard")


#==== aski chabot=====

st.markdown("### 📝 Enter a Post Title or Comment to Check for Bullying")
user_input = st.text_area("Enter text here:")

if user_input:
    prediction, confidence = is_bullying_model(user_input)
    if prediction == 1:
        st.success(f"✅ This is likely **cyberbullying** (Confidence: {confidence:.2%})")
    else:
        st.info(f"🔍 This doesn't appear to be bullying (Confidence: {confidence:.2%})")

# === Top Filters ===
st.subheader("📅 Select Date Range")
min_date = df['timestamp_utc'].min().date()
max_date = df['timestamp_utc'].max().date()
date_range = st.slider(
    "Filter by Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# === Sidebar Filters ===
st.sidebar.header("🔍 Filter Options")
platforms = st.sidebar.multiselect("Platforms", df['Platform'].unique(), default=df['Platform'].unique())
subreddits = st.sidebar.multiselect("Subreddits", df['Subreddit'].unique(), default=df['Subreddit'].unique())
bullying_only = st.sidebar.checkbox("Show only bullying posts")

# === Filtering Data ===
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (
    (df['timestamp_utc'].dt.date >= start_date.date()) &
    (df['timestamp_utc'].dt.date <= end_date.date()) &
    (df['Platform'].isin(platforms)) &
    (df['Subreddit'].isin(subreddits))
)
if bullying_only:
    mask &= df['Bullying'] == 1
filtered_df = df[mask]

# === KPIs ===
st.markdown("### 📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", len(filtered_df))
col2.metric("% Bullying Posts", f"{(filtered_df['Bullying'].mean()) * 100:.1f}%")
col3.metric("Most Active Subreddit", filtered_df['Subreddit'].mode().values[0] if not filtered_df.empty else "N/A")
col4.metric("Top Platform", filtered_df['Platform'].mode().values[0] if not filtered_df.empty else "N/A")

# === Bullying Posts by Date ===
st.subheader("1. 📅 Number of Bullying Posts by Date")
bully_by_date = filtered_df[filtered_df['Bullying'] == 1].groupby(filtered_df['timestamp_utc'].dt.date).size().reset_index(name='Bullying Posts')
fig1 = px.bar(bully_by_date, x='timestamp_utc', y='Bullying Posts', title='Bullying Posts Over Time')
st.plotly_chart(fig1, use_container_width=True)

# === Bullying vs Non-Bullying by Subreddit ===
st.subheader("2. ⚖️ Bullying vs Non-Bullying by Subreddit")
bully_by_subreddit = filtered_df.groupby(['Subreddit', 'Bullying']).size().reset_index(name='Count')
fig2 = px.bar(bully_by_subreddit, x='Subreddit', y='Count', color='Bullying', barmode='group')
st.plotly_chart(fig2, use_container_width=True)

# === Average Engagement ===
st.subheader("3. 📊 Avg Comments & Posts by Bullying Status")
agg_df = filtered_df.groupby('Bullying').agg({'Comments': 'mean', 'id': 'count'}).rename(columns={'id': 'Total Posts'}).reset_index()
agg_df['Bullying'] = agg_df['Bullying'].map({0: 'Non-Bullying', 1: 'Bullying'})
fig3 = px.bar(agg_df.melt(id_vars='Bullying', var_name='Metric', value_name='Average'),
              x='Bullying', y='Average', color='Metric', barmode='group')
st.plotly_chart(fig3, use_container_width=True)

# === Top Subreddits of Month ===
st.subheader("4. 🏆 Top 5 Subreddits of the Month")
month = st.selectbox("Select Month", sorted(df['timestamp_utc'].dt.strftime("%Y-%m").unique(), reverse=True))
top_subs = filtered_df[filtered_df['timestamp_utc'].dt.strftime("%Y-%m") == month].groupby('Subreddit').size().nlargest(5).reset_index(name='Post Count')
fig4 = px.bar(top_subs, x='Subreddit', y='Post Count', title=f'Top 5 Subreddits in {month}')
st.plotly_chart(fig4, use_container_width=True)

# === Word Cloud ===
st.subheader("5. 🧠 Word Cloud of Topics")
if not filtered_df.empty and filtered_df['Topic'].notna().any():
    wordcloud_data = ' '.join(filtered_df['Topic'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
else:
    st.warning("No topic data available for word cloud.")

# === Optional: Show raw data ===
if st.checkbox("Show raw data"):
    st.write(filtered_df.head(20))
