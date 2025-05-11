from fastapi import FastAPI, HTTPException
from src.utils.gemini_helper import generate_response
from src.utils.product_service import get_products_for_query
from config.requests import UserQuery, ChatResponse
import uvicorn

app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
async def chat(query: UserQuery):
    try:
        top_products = get_products_for_query(query.text)
        return generate_response(query.text,top_products)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)