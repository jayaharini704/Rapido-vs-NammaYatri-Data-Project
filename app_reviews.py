from google_play_scraper import reviews_all
import pandas as pd

def get_reviews(app_id, app_name):
    print(f"\nFetching reviews for {app_name}...")
    reviews = reviews_all(app_id, sleep_milliseconds=0, lang='en', country='in')
    
    if not reviews:
        print(f"⚠️ No reviews returned for {app_name}. Skipping.")
        return pd.DataFrame()  # empty DataFrame

    df = pd.DataFrame(reviews)
    
    if df.empty:
        print(f"⚠️ DataFrame is empty for {app_name}.")
        return pd.DataFrame()

    df["app"] = app_name

    return df[["app", "content", "score", "at"]]


rapido_reviews = get_reviews("com.rapido.passenger", "Rapido")
namma_reviews = get_reviews("in.juspay.nammayatri", "Namma Yatri")

all_reviews = pd.concat([rapido_reviews, namma_reviews], ignore_index=True)
print(f"✅ Total reviews fetched: {len(all_reviews)}")
all_reviews.to_csv("data/app_reviews.csv", index=False)