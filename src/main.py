from fastapi import FastAPI, Depends
from routers import patient_router, predict_router, user_router
from services.auth_service import validate_token
from contextlib import asynccontextmanager
import httpx, asyncio, os

RENDER_APP_URL = os.getenv("RENDER_APP_URL")

async def ping_self():
    url = RENDER_APP_URL
    if not url:
        print("Skipping self-ping: RENDER_APP_URL not set")
        return
    if "onrender.com" not in url:
        print("Skipping self-ping: Not running on Render")
        return
    health_url = f"{url.rstrip('/')}/health"

    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(health_url)
                print(f"Pinged self: ({health_url}): {r.status_code}")
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
app.include_router(predict_router.router, prefix="/api/predict", tags=["Heart failure prediction"], dependencies=[Depends(validate_token)])
app.include_router(patient_router.router, prefix="/api/patients", tags=["Patients"], dependencies=[Depends(validate_token)])

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}