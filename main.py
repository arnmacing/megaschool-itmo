from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from schemas.request import PredictionRequest, PredictionResponse
from tools.agent import predict

load_dotenv()

app = FastAPI()


@app.post("/api/request", response_model=PredictionResponse)
async def handle_request(body: PredictionRequest):
    """
    Эндпоинт для обработки POST запросов на /api/request.
    """
    try:
        return await predict(body)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка обработки запроса: {str(e)}"
        )
