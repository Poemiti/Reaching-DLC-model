from pathlib import Path
from typing import Literal, List
from pydantic import BaseModel, Field, validator
import yaml


class PathsConfig(BaseModel):
    data: Path
    labeling: Path
    model: Path
    evaluation: Path
    temporary: Path

    # @field_validator("*", mode="before")
    @validator("*")
    @classmethod
    def expand_paths(cls, v):
        return Path(v).expanduser().resolve()


class Config(BaseModel):
    project: str
    experimenter: str

    num_frames_per_video: int = Field(..., gt=0)
    extract_method: Literal["phash", "uniform"]

    optimizer: Literal["AdamW"]
    n_epochs: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    num_frames_for_train: int = Field(..., gt=0)

    paths: PathsConfig
    video_to_extract: List[Path]


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))