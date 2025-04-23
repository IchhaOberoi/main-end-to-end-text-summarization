from fastapi import FastAPI
import uvicorn
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textSummarizer.pipeline.prediction import PredictionPipeline

# Define Pydantic model for POST input
class RequestBody(BaseModel):
    text: str

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(data: RequestBody):
    try:
        obj = PredictionPipeline()
        result = obj.predict(data.text)
        return {"summary": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
