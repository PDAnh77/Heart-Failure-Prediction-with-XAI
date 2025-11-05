from fastapi import FastAPI
from schemas import patient_schema
from routers import patient_router, predict_router

app = FastAPI(title="Heart Disease Prediction API")

app.include_router(patient_router.router, prefix="/patients", tags=["Patients"])
app.include_router(predict_router.router, prefix="/predict", tags=["Heart failure prediction"])

@app.get("/")
def root():
    return {"message": "API đang chạy"}