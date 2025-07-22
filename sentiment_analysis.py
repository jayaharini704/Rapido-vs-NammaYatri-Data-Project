import pandas as pd
from textblob import TextBlob

# Load the dataset
df = pd.read_csv("data/app_reviews.csv")

# Drop rows with missing review content
df = df.dropna(subset=["content"])

# Function to get sentiment label
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["sentiment"] = df["content"].apply(get_sentiment)

# Remove any unnamed columns (commonly index columns accidentally saved)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Save the results
df.to_csv("data/app_reviews_with_sentiment.csv", index=False)

# Grouped sentiment counts per app
summary = df.groupby(["app", "sentiment"]).size().unstack().fillna(0)
print(summary)
