# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

from src.inference.pipeline import DetectionPipeline

app = FastAPI()

# Allow your front‑end to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the detector once
pipeline = DetectionPipeline()

@app.post("/detect/")
async def detect_file(file: UploadFile = File(...)):
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    contents = await file.read()
    temp_path.write_bytes(contents)

    result = pipeline.analyze(str(temp_path))

    temp_path.unlink()
    return JSONResponse(content=result)

# Serve all files in ./static under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root URL
@app.get("/", response_class=FileResponse)
async def root():
    return "static/index.html"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
