from fastapi import FastAPI, File, UploadFile

from .predictor import PredictionFunctor, Predictor

app = FastAPI()

predictor = PredictionFunctor(Predictor.mnist_model)


@app.post("/v1/predict")
async def predict(file: UploadFile = File(...)):
    """Run a prediction on the input image"""
    prediction = predictor(file.file)
    return {"prediction": prediction.shape, "predictorName": predictor.predictor}


@app.get("/v1/predictor")
async def get_predictor():
    """Get the name of the deployed predictor"""
    return {"name": predictor.predictor}
