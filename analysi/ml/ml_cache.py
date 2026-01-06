import os
import pickle

# 🔥 ALWAYS store inside app folder (Render safe)
BASE_DIR = os.path.dirname(__file__)   # analysi/ml
CACHE_FILE = os.path.join(BASE_DIR, "ml_results.pkl")


def save_ml_results(results):
    """
    Save ML results once (Algorithms page only)
    """
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)


def load_ml_results():
    """
    Load ML results everywhere else (Dashboard, Future, Profit)
    """
    if not os.path.exists(CACHE_FILE):
        return None

    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)
