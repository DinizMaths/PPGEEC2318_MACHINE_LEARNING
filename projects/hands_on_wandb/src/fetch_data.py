import os
import wandb

from dotenv import load_dotenv
from sklearn.datasets import fetch_openml

def fetch_data():
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame

    artifact = wandb.Artifact(
        name="adult_raw",
        type="raw_data",
        description="Raw adult dataset from OpenML",
    )

    os.makedirs("data", exist_ok=True)
    df.to_csv("./data/adult_raw.csv", index=False)

    wandb.init(
        project="hands_on_wandb",
        name="fetch_data",
        job_type="fetch_data",
        tags=["openml", "hands_on_wandb"]
    )

    artifact.add_file("./data/adult_raw.csv")
    wandb.log_artifact(artifact)

    wandb.finish()