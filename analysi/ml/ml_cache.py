import os
import pickle
from django.conf import settings

# ✅ Safe cache directory (auto-created)
CACHE_DIR = os.path.join(settings.BASE_DIR, "ml_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "ml_results.pkl")


def save_ml_results(results):
    # Create directory if missing
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)


def load_ml_results():
    if not os.path.exists(CACHE_FILE):
        return None

    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)
