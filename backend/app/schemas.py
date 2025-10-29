from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

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


class QuizQuestion(BaseModel):
    question_type: str
    question_prompt: str
    options: Optional[List[str]] = None
    correct_answer: Union[str, List[str]]
    answer_rubric: str
    hint: str
    answer_explanation: str


class QuizQuestionRequest(BaseModel):
    lecture_key: Optional[str] = None
    topic: Optional[str] = None
    question_type: Optional[str] = None


class QuizQuestionResponse(BaseModel):
    question: QuizQuestion
    context: str
    lecture_key: Optional[str] = None
    topic: Optional[str] = None


class QuizGradeRequest(BaseModel):
    question: QuizQuestion
    context: str
    user_answer: Union[str, List[str]]


class QuizGradeResponse(BaseModel):
    correct: bool
    score: float
    assessment: str
    good_points: List[str]
    bad_points: List[str]
