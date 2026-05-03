from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predicts import router as predict_router
from routes.charts import router as charts_router

app = FastAPI(
    title="CareerLens Pakistan API",
    description="AI-Powered Career & Salary Intelligence Platform",
    version="1.0.0"
)

# ─── CORS (React frontend connect hoga) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ROUTES ──────────────────────────────────────────────
app.include_router(predict_router, prefix="/api")
app.include_router(charts_router,  prefix="/api")

# ─── ROOT ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "CareerLens Pakistan API is running! 🚀"}