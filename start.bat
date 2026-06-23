@echo off
echo Starting FastAPI Backend...
start cmd /k "uvicorn server:app --reload --port 8000"

echo Starting React Frontend...
cd frontend
start cmd /k "npm run dev"
