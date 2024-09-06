from dataclasses import dataclass
from pathlib import Path
import pickle


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    STATUS_FILE: str
    all_schema: dict



@dataclass(frozen=True)
class DataTransfornmationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    train_data_path:Path
    test_data_path:Path
    model_name:str 
    target_column:str
    subsample: float
    n_estimators: int
    min_samples_split: int
    min_samples_leaf: int 
    max_depth: int
    learning_rate: float    


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:Path
    test_data_path:Path
    model_path:Path
    metrics_file_name:Path 
    target_column:str
    all_params:dict      