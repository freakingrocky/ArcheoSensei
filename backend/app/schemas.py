from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class UploadLectureRequest(BaseModel):
    course: Optional[str] = None

class QueryOptions(BaseModel):
    force_lecture_key: Optional[str] = None
    use_global: bool = True
    user_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    options: Optional[QueryOptions] = None

class MemorizeRequest(BaseModel):
    user_id: str
    text: str
