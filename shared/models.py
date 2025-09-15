from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Product(BaseModel):
    id: str
    product_name: str
    gender: str | None = None
    brand: str | None = None
    description: str | None = None
    price: float | None = None
    color: str | None = None