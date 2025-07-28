import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Set Streamlit page config
st.set_page_config(page_title="Rapido vs Namma Yatri - Auto Reviews Analysis", layout="wide")
st.title("🚖 Rapido vs Namma Yatri - Reviews Dashboard (Auto Only)")

# Load the data
df = pd.read_csv("data/app_reviews_auto_sentiment.csv")
df["at"] = pd.to_datetime(df["at"], errors="coerce", dayfirst=True)
df.dropna(subset=["at"], inplace=True)

# Pie chart - overall sentiment split for each app
st.subheader("1️⃣ Sentiment Distribution per App")
fig1 = px.histogram(df, x="sentiment", color="app", barmode="group",
                    title="Sentiment Comparison: Rapido vs Namma Yatri",
                    text_auto=True, color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig1, use_container_width=True)
st.markdown("""
**Insight 🧠:** Namma Yatri has a higher proportion of **positive auto reviews**, 
while Rapido shows a **higher number of negative sentiments**, suggesting potential issues 
with auto service quality or pricing.
""")

# Pie chart breakdown for selected app
st.subheader("2️⃣ Sentiment Breakdown - Pie Chart")
selected_app = st.selectbox("Choose an app:", df["app"].unique())
pie_data = df[df["app"] == selected_app]["sentiment"].value_counts()
fig_pie = px.pie(pie_data, values=pie_data.values, names=pie_data.index,
                 title=f"Sentiment Breakdown for {selected_app}",
                 color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig_pie, use_container_width=True)
st.markdown(f"""
**Insight 🧠:** Among all {selected_app} reviews, the majority are 
**{pie_data.idxmax()}**, showing that most users felt **\"{pie_data.idxmax()}\"** about the auto service.
""")

# Cumulative review count
st.subheader("3️⃣ Cumulative Auto Review Growth Over Time")
df_sorted = df.sort_values("at")
df_sorted["count"] = 1
df_sorted["cumulative_count"] = df_sorted.groupby("app")["count"].cumsum()
fig2 = px.line(df_sorted, x="at", y="cumulative_count", color="app",
               title="Cumulative Reviews Over Time",
               color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
**Insight 🧠:** Namma Yatri shows a **steeper cumulative growth** after launch, 
indicating **rapid adoption** of their auto booking service. Rapido’s growth is more **gradual**.
""")

# Monthly trend
st.subheader("4️⃣ Monthly Auto Reviews Trend")
df["month"] = df["at"].dt.to_period("M").astype(str)
monthly = df.groupby(["month", "app"]).size().reset_index(name="count")
fig3 = px.line(monthly, x="month", y="count", color="app",
               title="Monthly Reviews Trend",
               markers=True, color_discrete_sequence=px.colors.qualitative.Dark24)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("""
**Insight 🧠:** Monthly review spikes correlate with **app updates or public campaigns**. 
Namma Yatri sees review surges around early 2023, possibly linked to government-backed promotion.
""")

# Sentiment over time
st.subheader("5️⃣ Sentiment Over Time")
sentiment_trend = df.groupby(["month", "app", "sentiment"]).size().reset_index(name="count")
fig4 = px.line(sentiment_trend, x="month", y="count", color="sentiment", facet_col="app",
               title="Sentiment Over Time per App",
               markers=True, color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig4, use_container_width=True)
st.markdown("""
**Insight 🧠:** Sentiments for Namma Yatri became more **positive over time**, 
while Rapido shows **fluctuations**, indicating varying user satisfaction.
""")


# Reviews per Year
st.subheader("6️⃣ Reviews per Year")
df["year"] = df["at"].dt.year
yearwise = df.groupby(["year", "app"]).size().reset_index(name="count")
fig7 = px.bar(yearwise, x="year", y="count", color="app",
              barmode="group", title="Reviews per Year per App",
              text_auto=True, color_discrete_sequence=px.colors.qualitative.Prism)
st.plotly_chart(fig7, use_container_width=True)
st.markdown("""
**Insight 🧠:** Namma Yatri reviews pick up sharply from 2023 onwards, 
while Rapido maintains a steady stream across years.
""")

# Total auto review counts
st.subheader("7️⃣ Total Auto Reviews")
total_reviews = df["app"].value_counts().reset_index()
total_reviews.columns = ["App", "Total Reviews"]
fig8 = px.bar(total_reviews, x="App", y="Total Reviews", color="App",
              text_auto=True, title="Total Auto Reviews per App",
              color_discrete_sequence=px.colors.qualitative.Alphabet)
st.plotly_chart(fig8, use_container_width=True)
st.markdown("""
**Insight 🧠:** Rapido has more total auto reviews overall, but distribution over time 
and sentiment trends offer **deeper comparative insights**.
""")
