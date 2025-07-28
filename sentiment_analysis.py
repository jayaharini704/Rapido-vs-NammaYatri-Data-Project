import pandas as pd
from textblob import TextBlob

df = pd.read_csv("data/app_reviews.csv")


df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


df = df.dropna(subset=["content"])


df["content_lower"] = df["content"].str.lower()


auto_keywords = ["auto", "rickshaw", "meter", "three wheeler", "autorickshaw"]
bike_keywords = ["bike", "rider", "helmet", "two wheeler"]
cab_keywords = ['ac cab', 'cab', 'taxi', 'car', 'vehicle', 'sedan', 'suv', 
                'taxi driver', 'cab driver', 'xl cab', 'xl premium', 'non ac cab']

df["is_auto_review"] = df["content_lower"].apply(
    lambda x: any(keyword in x for keyword in auto_keywords)
)

df_auto = df[df["is_auto_review"]]
df_auto = df_auto[
    ~df_auto["content_lower"].str.contains('|'.join(bike_keywords + cab_keywords))
]


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df_auto["sentiment"] = df_auto["content"].apply(get_sentiment)

df_auto.to_csv("data/app_reviews_auto_sentiment.csv", index=False)


summary = df_auto.groupby(["app", "sentiment"]).size().unstack().fillna(0)


summary["Total Auto Reviews"] = summary.sum(axis=1)


print(summary)

summary.to_csv("data/app_reviews_auto_sentiment_summary.csv")