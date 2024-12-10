from pydantic import BaseModel


class PromptMessage(BaseModel):
    message: str
