import os
import joblib
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

def safe_load_joblib(file_name):
    try:
        return joblib.load(os.path.join(MODEL_DIR, file_name))
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")
        return None

tfidf = safe_load_joblib("tfidf_vectorizer.pkl")
voting_model = safe_load_joblib("voting_model.pkl")
stacked_meta_model = safe_load_joblib("stacked_meta_model.pkl")

def predict_text(text):
    results = {}
    if tfidf and voting_model:
        X_tfidf = tfidf.transform([text])
        results["Voting"] = int(voting_model.predict(X_tfidf)[0])
    if tfidf and stacked_meta_model:
        X_tfidf = tfidf.transform([text])
        results["Stacked"] = int(stacked_meta_model.predict(X_tfidf)[0])
    if not results:
        results["Error"] = "No model loaded."
    return results
