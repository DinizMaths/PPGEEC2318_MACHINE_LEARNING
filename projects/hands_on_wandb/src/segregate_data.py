import os
import wandb
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

def create_and_log_artifact(file_path, artifact_name, artifact_type, description):
    """Cria um artefato e o registra no wandb."""
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
    )
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)

def segregate_data():
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    run = wandb.init(
        project="hands_on_wandb",
        name="segregate_data",
        job_type="segregate_data",
        tags=["openml", "hands_on_wandb"]
    )

    artifact = run.use_artifact("hands_on_wandb/adult_preprocessed:latest")
    df = pd.read_csv(artifact.file())

    # Separar os dados em características e rótulos
    X = df.drop(columns=["income"])
    y = df["income"]

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    os.makedirs("data", exist_ok=True)

    # Salvar e registrar os dados de treinamento e teste
    X_train.to_csv("./data/X_train.csv", index=False)
    create_and_log_artifact("./data/X_train.csv", "X_train", "train_data", "Training data")

    X_test.to_csv("./data/X_test.csv", index=False)
    create_and_log_artifact("./data/X_test.csv", "X_test", "test_data", "Test data")

    y_train.to_csv("./data/y_train.csv", index=False)
    create_and_log_artifact("./data/y_train.csv", "y_train", "train_data", "Training data")

    y_test.to_csv("./data/y_test.csv", index=False)
    create_and_log_artifact("./data/y_test.csv", "y_test", "test_data", "Test data")

    wandb.finish()
