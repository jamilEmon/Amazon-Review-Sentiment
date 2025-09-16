import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow import keras
import warnings
from utils.preprocess import clean_text

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

def load_joblib_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")
        return None

# Load models
tfidf = load_joblib_model("tfidf_vectorizer.pkl")
voting_model = load_joblib_model("voting_model.pkl")
stacked_meta_model = load_joblib_model("stacked_meta_model.pkl")
tokenizer = load_joblib_model("tokenizer.pkl")

try:
    cnn_model = load_model(os.path.join(MODEL_DIR, "cnn_model.h5"), compile=False)
except Exception as e:
    print(f"Failed to load CNN Model: {e}")
    cnn_model = None

def predict_text(text):
    """Run predictions across all available models."""
    text = clean_text(text)
    results = {}

    def label(val):
        return "Positive" if val == 1 else "Negative"

    try:
        if tfidf is not None:
            X_tfidf = tfidf.transform([text])
            if voting_model:
                pred = int(voting_model.predict(X_tfidf)[0])
                results["Voting"] = label(pred)
            if stacked_meta_model:
                pred = int(stacked_meta_model.predict(X_tfidf)[0])
                results["Stacked"] = label(pred)
    except Exception as e:
        results["TFIDF Error"] = str(e)

    try:
        if tokenizer and cnn_model:
            seq = tokenizer.texts_to_sequences([text])
            padded = keras.utils.pad_sequences(seq, maxlen=200)
            pred = (cnn_model.predict(padded, verbose=0) > 0.5).astype("int")[0][0]
            results["CNN"] = label(pred)
    except Exception as e:
        results["CNN Error"] = str(e)

    if not results:
        results["Error"] = "No model loaded."
    return results
