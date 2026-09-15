from flask import Flask, request, jsonify
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load recommendation model components
tfidf = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
recommendation_df = joblib.load("recommendation_data.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Personalized Job Recommendation API is running"
    })


@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.get_json()

    user_query = data.get("user_query", "")
    country = data.get("country", "All")
    job_type = data.get("job_type", "All")
    top_n = int(data.get("top_n", 10))

    # Validate user query
    if not user_query.strip():
        return jsonify({
            "error": "user_query is required"
        }), 400

    # Convert user query into TF-IDF
    query_vector = tfidf.transform([user_query])

    # Calculate similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = recommendation_df.copy()
    results["similarity_score"] = similarity_scores

    # Country filter
    if country != "All":
        results = results[
            results["country"].fillna("").str.lower()
            == country.lower()
        ]

    # Job type filter
    if job_type == "Hourly":
        results = results[
            results["is_hourly"] == True
        ]

    elif job_type == "Fixed Price":
        results = results[
            results["is_hourly"] == False
        ]

    # Sort by similarity
    results = results.sort_values(
        by="similarity_score",
        ascending=False
    )

    # Remove duplicate titles ignoring capitalization
    results["title_clean"] = (
        results["title"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    results = results.drop_duplicates(
        subset=["title_clean"],
        keep="first"
    )

    # Top recommendations
    results = results.head(top_n)

    recommendations = []

    for _, row in results.iterrows():

        recommendations.append({
            "title": row["title"],
            "country": row["country"],
            "is_hourly": bool(row["is_hourly"]),
            "hourly_low": (
                None if pd.isna(row["hourly_low"])
                else float(row["hourly_low"])
            ),
            "hourly_high": (
                None if pd.isna(row["hourly_high"])
                else float(row["hourly_high"])
            ),
            "budget": (
                None if pd.isna(row["budget"])
                else float(row["budget"])
            ),
            "link": row["link"],
            "similarity_score": round(
                float(row["similarity_score"]), 4
            )
        })

    return jsonify({
        "user_query": user_query,
        "country": country,
        "job_type": job_type,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)