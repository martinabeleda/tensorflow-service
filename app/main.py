from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.predictor import PredictionFunctor, Predictor

app = FastAPI()

predictor = PredictionFunctor(Predictor.mnist_dropout)


@app.post("/v1/predict")
async def predict(file: UploadFile = File(...)):
    """Run a prediction on the input image"""
    prediction, uncertainty = predictor(file.file)
    return JSONResponse(
        content={
            "prediction": prediction.tolist(),
            "uncertainty": uncertainty.tolist(),
            "predictorName": predictor.predictor,
        }
    )


@app.get("/v1/predictor")
async def get_predictor():
    """Get the name of the deployed predictor"""
    return JSONResponse(content={"name": predictor.predictor})
