from pydantic import BaseModel


class TrainingConfig(BaseModel):
    model_checkpoint_path: str = "./model.pth"
    data_root_dir: str = "./data"
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    num_epochs: int = 1000
    patience: int = 15
    classes: list = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
