import joblib
from db.database import supabase

_pipeline_instance = None

BUCKET_NAME = "heart-prediction-models"
MODEL_FILENAME = "model_predict.pkl"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"

def load_model_startup():
    global _pipeline_instance
    print(f"Connecting Storage to download {MODEL_FILENAME}...")
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(MODEL_FILENAME)
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(data)
            
        _pipeline_instance = joblib.load(LOCAL_MODEL_PATH)
        print(f"Download complete.")
    except Exception as e:
        print(f"Unable to download model from Supabase.")
        print(f"Error details: {str(e)}")
        raise e

def get_pipeline():
    if _pipeline_instance is None:
        # Fallback: Nếu chưa load thì load ngay lập tức
        load_model_startup()
    return _pipeline_instance