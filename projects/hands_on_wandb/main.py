import os

from src.fetch_data import fetch_data
from src.preprocess_data import preprocess_data
from src.segregate_data import segregate_data
from src.preparate_data import preparate_data

if __name__ == "__main__":
    fetch_data()
    preprocess_data()

    os.system("pytest -v ./src/test_data.py")

    segregate_data()
    preparate_data()