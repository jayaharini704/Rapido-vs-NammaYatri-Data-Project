import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs("plots", exist_ok=True)

# Load data
df = pd.read_csv("data/app_reviews_auto_sentiment.csv")

# Clean Unnamed columns (if any)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Convert date column
df["at"] = pd.to_datetime(df["at"], errors="coerce", dayfirst=True)

# Drop rows with invalid date
df = df.dropna(subset=["at"])

# --- 1. Sentiment Distribution Per App ---
plt.figure(figsize=(8, 5))
sentiment_counts = df.groupby(["app", "sentiment"]).size().reset_index(name="count")
sns.barplot(x="app", y="count", hue="sentiment", data=sentiment_counts)
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%d', label_type='edge', fontsize=8)
plt.title("Sentiment Distribution by App (Auto-specific Reviews)")
plt.ylabel("Number of Reviews")
plt.xlabel("App")
plt.tight_layout()
plt.savefig("plots/sentiment_distribution.png")
plt.close()

# --- 2. Cumulative Review Growth Over Time ---
df_sorted = df.sort_values("at")
df_sorted["count"] = 1
df_sorted["cumulative"] = df_sorted.groupby("app")["count"].cumsum()
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_sorted, x="at", y="cumulative", hue="app")
plt.title("Cumulative Growth of Auto Reviews Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Reviews")
plt.tight_layout()
plt.savefig("plots/cumulative_reviews.png")
plt.close()

# --- 3. Monthly Review Count by App ---
df["year_month"] = df["at"].dt.to_period("M").astype(str)
monthly_counts = df.groupby(["year_month", "app"]).size().reset_index(name="count")
plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_counts, x="year_month", y="count", hue="app", marker="o")
plt.title("Monthly Auto Reviews by App")
plt.xticks(rotation=45)
plt.ylabel("Review Count")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("plots/monthly_reviews.png")
plt.close()

# --- 4. Sentiment Percentage Per App ---
total_per_app = df.groupby("app").size().reset_index(name="total")
sentiment_pct = sentiment_counts.merge(total_per_app, on="app")
sentiment_pct["percentage"] = (sentiment_pct["count"] / sentiment_pct["total"]) * 100

plt.figure(figsize=(8, 5))
sns.barplot(x="app", y="percentage", hue="sentiment", data=sentiment_pct)
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=8)
plt.title("Sentiment % Distribution by App")
plt.ylabel("Percentage")
plt.xlabel("App")
plt.tight_layout()
plt.savefig("plots/sentiment_percentage.png")
plt.close()

# --- 5. Sentiment Trend Over Time (Line) ---
sentiment_time = df.groupby([df["at"].dt.to_period("M").astype(str), "app", "sentiment"]).size().reset_index(name="count")
plt.figure(figsize=(12, 5))
sns.lineplot(data=sentiment_time, x="at", y="count", hue="sentiment", style="app", marker="o")
plt.title("Sentiment Trend Over Time (Monthly Auto Reviews)")
plt.xticks(rotation=45)
plt.ylabel("Review Count")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("plots/sentiment_trend_monthly.png")
plt.close()

# --- 6. Pie Chart of Sentiment Split (Per App) ---
for app_name in df["app"].unique():
    plt.figure(figsize=(5, 5))
    app_sentiment = df[df["app"] == app_name]["sentiment"].value_counts()
    plt.pie(app_sentiment, labels=app_sentiment.index, autopct="%1.1f%%", startangle=140)
    plt.title(f"{app_name} - Sentiment Split (Auto Reviews)")
    plt.tight_layout()
    plt.savefig(f"plots/{app_name.lower().replace(' ', '_')}_sentiment_pie.png")
    plt.close()

# --- 7. Word Count Distribution (Length of Review Text) ---
df["word_count"] = df["content"].astype(str).apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="word_count", hue="app", bins=30, element="step", kde=True)
plt.title("Distribution of Word Count in Auto Reviews")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("plots/word_count_distribution.png")
plt.close()

# --- 8. Sentiment Heatmap ---
pivot = sentiment_counts.pivot(index="app", columns="sentiment", values="count").fillna(0)
plt.figure(figsize=(6, 4))
sns.heatmap(pivot, annot=True, fmt="g", cmap="YlGnBu")
plt.title("Sentiment Counts by App")
plt.tight_layout()
plt.savefig("plots/sentiment_heatmap.png")
plt.close()

