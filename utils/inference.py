import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow import keras
import warnings

warnings.filterwarnings("ignore")  # ignore sklearn warnings

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
tokenizer = safe_load_joblib("tokenizer.pkl")

try:
    cnn_model = load_model(os.path.join(MODEL_DIR, "cnn_model.h5"), compile=False)
except Exception as e:
    print(f"Failed to load CNN model: {e}")
    cnn_model = None

def predict_text(text):
    results = {}
    if tfidf and voting_model:
        X_tfidf = tfidf.transform([text])
        results["Voting"] = int(voting_model.predict(X_tfidf)[0])
    if tfidf and stacked_meta_model:
        X_tfidf = tfidf.transform([text])
        results["Stacked"] = int(stacked_meta_model.predict(X_tfidf)[0])
    if tokenizer and cnn_model:
        seq = tokenizer.texts_to_sequences([text])
        padded = keras.utils.pad_sequences(seq, maxlen=200)
        pred = (cnn_model.predict(padded, verbose=0) > 0.5).astype("int")[0][0]
        results["CNN"] = int(pred)
    if not results:
        results["Error"] = "No model loaded."
    return results
