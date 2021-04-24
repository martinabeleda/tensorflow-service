from fastapi import FastAPI, File, UploadFile

from .predictor import PredictionFunctor, Predictor

app = FastAPI()

predictor = PredictionFunctor(Predictor.mnist_model)


@app.post("/v1/predict")
async def predict(file: UploadFile = File(...)):
    prediction = predictor(file.file)
    return {"prediction": prediction.shape, "predictor": str(predictor.predictor)}
