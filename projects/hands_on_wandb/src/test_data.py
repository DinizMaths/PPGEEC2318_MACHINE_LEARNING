import os
import wandb
import pytest
import pandas as pd
from dotenv import load_dotenv

@pytest.fixture(scope="session", autouse=True)
def wandb_setup():
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    run = wandb.init(
        project="hands_on_wandb",
        name="test_data",
        job_type="test_data",
        tags=["openml", "hands_on_wandb"],
    )

    yield run

    run.finish()

@pytest.fixture(scope="session")
def data(wandb_setup):
    artifact = wandb_setup.use_artifact("hands_on_wandb/adult_preprocessed:latest")
    df = pd.read_csv(artifact.file())
    return df

def test_data_length(data, wandb_setup):
    length = len(data)
    wandb.log({"data_length": length})  # Log do comprimento dos dados
    assert length > 1000

def test_number_of_columns(data, wandb_setup):
    num_columns = data.shape[1]
    wandb.log({"num_columns": num_columns})  # Log do número de colunas
    assert num_columns == 15
