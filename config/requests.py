from pydantic import BaseModel

class UserQuery(BaseModel):
    text: str

class ProductOut(BaseModel):
    id: str
    title: str
    price: str
    discount: str
    url: str

class ChatResponse(BaseModel):
    response_type: str
    products: list[ProductOut]
    answer: str