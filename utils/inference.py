import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow import keras
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")  # Ensure this matches your project structure

# Load models safely
def safe_load_joblib(filename):
    try:
        return joblib.load(os.path.join(MODEL_DIR, filename))
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return None

try:
    tfidf = safe_load_joblib("tfidf_vectorizer.pkl")
    voting_model = safe_load_joblib("voting_model.pkl")
    stacked_meta_model = safe_load_joblib("stacked_meta_model.pkl")
    tokenizer = safe_load_joblib("tokenizer.pkl")
except Exception as e:
    print(f"Error loading models: {e}")

try:
    cnn_model = load_model(os.path.join(MODEL_DIR, "cnn_model.h5"), compile=False)
except Exception as e:
    print(f"Failed to load CNN model: {e}")
    cnn_model = None

from .preprocess import clean_text

def predict_text(text):
    text = clean_text(text)
    results = {}

    if tfidf and voting_model:
        X_tfidf = tfidf.transform([text])
        try:
            results["Voting"] = int(voting_model.predict(X_tfidf)[0])
        except Exception as e:
            results["Voting"] = f"Error: {e}"

    if tfidf and stacked_meta_model:
        X_tfidf = tfidf.transform([text])
        try:
            results["Stacked"] = int(stacked_meta_model.predict(X_tfidf)[0])
        except Exception as e:
            results["Stacked"] = f"Error: {e}"

    if tokenizer and cnn_model:
        seq = tokenizer.texts_to_sequences([text])
        padded = keras.utils.pad_sequences(seq, maxlen=200)
        try:
            pred = (cnn_model.predict(padded, verbose=0) > 0.5).astype("int")[0][0]
            results["CNN"] = int(pred)
        except Exception as e:
            results["CNN"] = f"Error: {e}"

    if not results:
        results["Error"] = "No models loaded."

    return results
