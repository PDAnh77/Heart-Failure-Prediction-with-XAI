from fastapi import FastAPI, Depends
from routers import patient_router, predict_router, user_router
from services.auth import validate_token
from contextlib import asynccontextmanager
import httpx, asyncio, os

RENDER_APP_URL = os.getenv("RENDER_APP_URL")

async def ping_self():
    if not RENDER_APP_URL or "onrender.com" not in RENDER_APP_URL:
        print("Skipping self-ping")
        return
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(RENDER_APP_URL)
                print(f"Pinged self: {r.status_code}")
        except Exception as e:
            print(f"Error pinging self: {e}")
        await asyncio.sleep(10 * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(ping_self())
    yield
    task.cancel() # Hủy task khi API tắt

app = FastAPI(title="Heart Disease Prediction API", openapi_url="/api/openapi.json", docs_url="/docs", lifespan=lifespan)

app.include_router(user_router.router, prefix="/api/user", tags=["User"])
app.include_router(predict_router.router, prefix="/api/predict", tags=["Heart failure prediction"])
app.include_router(patient_router.router, prefix="/api/patients", tags=["Patients"], dependencies=[Depends(validate_token)])

@app.get("/")
def root():
    return {"message": "API is running"}