from pydantic import BaseModel


class TrainingConfig(BaseModel):
    batch_size: int = 64
