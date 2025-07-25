import numpy as np
import pandas as pd
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def predict_er(avg_likes, avg_comments, followers, caption, hashtags):

    # ---------------------------- Load Trained Model ----------------------------
    model_filename = 'path_to_random_forest_engagement_model.pkl'
    model = joblib.load(model_filename)
    print("✅ Model loaded from disk.")

    # ---------------------------- Load Hashtag Popularity Data ----------------------------
    with open("path_to_tags.json", "r") as file:
        hashtag_popularity_data = json.load(file)

    # Convert to dictionary with hashtag format
    hashtag_popularity = {f"#{tag}": popularity for tag, popularity in hashtag_popularity_data}

    # ---------------------------- Hashtag Quality Scoring Function ----------------------------
    def hashtag_quality(hashtags, caption):
        if not isinstance(hashtags, str) or len(hashtags) == 0:
            return 0

        hashtag_list = hashtags.split()

        # Popularity Score
        popularity_score = sum(hashtag_popularity.get(tag, 0) for tag in hashtag_list) / max(len(hashtag_list), 1)

        # Relevance Score via TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([caption] + hashtag_list)
        caption_vector = tfidf_matrix[0]
        hashtag_vectors = tfidf_matrix[1:]
        relevance_score = np.mean([
            cosine_similarity(caption_vector, hashtag_vector)[0][0] for hashtag_vector in hashtag_vectors
        ])

        # Final Score (weighted)
        quality_score = (0.7 * popularity_score + 0.3 * relevance_score) / 10

        # Penalty for fewer than 5 hashtags
        penalty_factor = min(1, len(hashtag_list) / 5)
        adjusted_quality_score = quality_score * penalty_factor

        return min(adjusted_quality_score, 100)

    # ---------------------------- Engagement Rate Classifier ----------------------------
    def classify_engagement_rate(rate):
        if rate < 20:
            return "Low"
        elif rate < 40:
            return "Moderate"
        elif rate <= 60:
            return "High"
        else:
            return "Viral"

    ### ---------------------------- Sample Post Input ----------------------------
    post = {
        'likes': avg_likes,
        'comments': avg_comments,
        'followers': followers,
        'caption': caption,
        'hashtags': hashtags
    }

    # ---------------------------- Run Prediction ----------------------------
    # Step 1: Hashtag Quality
    hashtag_quality_score = hashtag_quality(post['hashtags'], post['caption'])
    print(f"🌟 Hashtag Quality Score: {hashtag_quality_score:.2f}")

    # Step 2: Predict Engagement Rate
    sample_input_df = pd.DataFrame([post], columns=['likes', 'comments', 'followers'])
    predicted_engagement_rate = model.predict(sample_input_df)[0]

    # Step 3: Adjust Engagement Rate using Hashtag Quality
    adjusted_engagement_rate = predicted_engagement_rate * (1 + (hashtag_quality_score / 100))
    adjusted_engagement_rate = min(adjusted_engagement_rate, 100)

    # Step 4: Classify
    engagement_category = classify_engagement_rate(adjusted_engagement_rate)

    return {
        "adjusted_engagement_rate": round(adjusted_engagement_rate, 2),
        "engagement_category": engagement_category
    }
