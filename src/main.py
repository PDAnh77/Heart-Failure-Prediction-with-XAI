from fastapi import FastAPI, Depends
from routers import patient_router, predict_router, user_router
from services.auth import validate_token

app = FastAPI(title="Heart Disease Prediction API", openapi_url="/api/openapi.json", docs_url="/docs")

app.include_router(user_router.router, prefix="/api/user", tags=["User"])
app.include_router(predict_router.router, prefix="/api/predict", tags=["Heart failure prediction"])
app.include_router(patient_router.router, prefix="/api/patients", tags=["Patients"], dependencies=[Depends(validate_token)])

@app.get("/")
def root():
    return {"message": "API is running"}