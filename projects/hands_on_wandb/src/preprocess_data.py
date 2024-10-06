import os
import wandb

import pandas as pd

from dotenv import load_dotenv


def preprocess_data():
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    run = wandb.init(
        project="hands_on_wandb",
        name="preprocess_data",
        job_type="preprocess_data",
        tags=["openml", "hands_on_wandb"]
    )

    artifact = run.use_artifact("hands_on_wandb/adult_raw:latest")
    df = pd.read_csv(artifact.file())

    df.drop_duplicates(inplace=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("./data/adult_preprocessed.csv", index=False)

    artifact = wandb.Artifact(
        name="adult_preprocessed",
        type="preprocessed_data",
        description="Preprocessed adult dataset from OpenML",
    )

    artifact.add_file("./data/adult_preprocessed.csv")
    wandb.log_artifact(artifact)

    wandb.finish()
