from fastapi import FastAPI
from routes import router

app = FastAPI(title="Face Verification System")

app.include_router(router)
