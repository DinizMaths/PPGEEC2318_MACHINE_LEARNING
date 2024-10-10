import os
import wandb

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # Remove white space in categorical features
        df = df.apply(lambda row: row.str.strip())

        # customize feature?
        # How can I identify what needs to be modified? EDA!!!!
        if self.new_features:

            # minimize the cardinality of native_country feature
            # check cardinality using df.native_country.unique()
            df.loc[df['native-country'] != 'United-States', 'native-country'] = 'non_usa'

            # replace ? with Unknown
            edit_cols = ['native-country', 'occupation', 'workclass']
            for col in edit_cols:
                df.loc[df[col] == '?', col] = 'unknown'

            # decrease the cardinality of education feature
            hs_grad = ['HS-grad', '11th', '10th', '9th', '12th']
            elementary = ['1st-4th', '5th-6th', '7th-8th']
            # replace
            df['education'] = df['education'].replace(to_replace=hs_grad, value='HS-grad')
            df['education'] = df['education'].replace(to_replace=elementary, value='elementary_school',)

            # adjust marital_status feature
            married = ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
            separated = ['Separated', 'Divorced']

            # replace
            df['marital-status'] = df['marital-status'].replace(to_replace=married, value='Married')
            df['marital-status'] = df['marital-status'].replace(to_replace=separated, value='Separated')

            # adjust workclass feature
            self_employed = ['Self-emp-not-inc', 'Self-emp-inc']
            govt_employees = ['Local-gov', 'State-gov', 'Federal-gov']

            # replace elements in list.
            df['workclass'] = df['workclass'].replace(to_replace=self_employed, value='Self_employed')
            df['workclass'] = df['workclass'].replace(to_replace=govt_employees, value='Govt_employees')

        # update column names
        self.colnames = df.columns

        return df

# transform numerical features
class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes a model parameter as its argument
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model=0, colnames=None):
        self.model = model
        self.colnames = colnames
        self.scaler = None

    # Fit is used only to learn statistical about Scalers
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)
        # minmax
        if self.model == 0:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df)
        # standard scaler
        elif self.model == 1:
            self.scaler = StandardScaler()
            self.scaler.fit(df)
        return self

    # return columns names after transformation
    def get_feature_names_out(self):
        return self.colnames

    # Transformer method we wrote for this transformer
    # Use fitted scalers
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # update columns name
        self.colnames = df.columns.tolist()

        # minmax
        if self.model == 0:
            # transform data
            df = self.scaler.transform(df)
        elif self.model == 1:
            # transform data
            df = self.scaler.transform(df)
        else:
            df = df.values

        return df

def load_data_from_wandb(artifact_name, artifact_type):
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    run = wandb.init(
        project="hands_on_wandb",
        name=f"load_data_{artifact_type}",
        job_type="load_data"
    )

    # Baixar o artefato correspondente
    artifact = run.use_artifact(f"hands_on_wandb/{artifact_name}:latest")
    artifact_dir = artifact.download()

    # Carregar o arquivo CSV baixado em um DataFrame
    df = pd.read_csv(os.path.join(artifact_dir, f"{artifact_name}.csv"))

    return df
    
def preparate_data():
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    run = wandb.init(
        project="hands_on_wandb",
        name="preparate_data",
        job_type="preparate_data",
        tags=["openml", "hands_on_wandb"]
    )

    X_train = load_data_from_wandb("X_train", "train_data")
    X_val = load_data_from_wandb("X_val", "val_data")
    y_train = load_data_from_wandb("y_train", "train_data")
    y_val = load_data_from_wandb("y_val", "val_data")

    # Categrical features to pass down the categorical pipeline
    categorical_features = X_train.select_dtypes("object").columns.to_list()

    # Numerical features to pass down the numerical pipeline
    numerical_features = X_train.select_dtypes("int64").columns.to_list()

    categorical_pipeline = Pipeline(
        steps=[
            ("cat_selector", FeatureSelector(categorical_features)),
            ("imputer_cat", SimpleImputer(strategy="most_frequent")),
            ("cat_transformer", CategoricalTransformer(colnames=categorical_features)),
            ("cat_encoder", OneHotEncoder(sparse_output=False, drop="first"))
        ]
    )

    numerical_model = 0
    numerical_pipeline = Pipeline(
        steps=[
            ("num_selector", FeatureSelector(numerical_features)),
            ("imputer_num", SimpleImputer(strategy="median")),
            ("num_transformer", NumericalTransformer(numerical_model, colnames=numerical_features))
        ]
    )

    full_pipeline_preprocessing = FeatureUnion(
        transformer_list=[
            ("cat_pipeline", categorical_pipeline),
            ("num_pipeline", numerical_pipeline)
        ]
    )

    pipe = Pipeline(
        steps=[
            ("full_pipeline", full_pipeline_preprocessing),
            ("classifier", DecisionTreeClassifier())
        ]
    )

    pipe.fit(X_train, y_train)

    predict_bias = pipe.predict(X_train)

    fbeta = fbeta_score(y_train, predict_bias, beta=1, pos_label=">50K", zero_division=1)
    precision = precision_score(y_train, predict_bias, pos_label=">50K", zero_division=1)
    recall = recall_score(y_train, predict_bias, pos_label=">50K", zero_division=1)
    acc = accuracy_score(y_train, predict_bias)

    if run:
        run.summary["Acc [train]"] = acc
        run.summary["Precision [train]"] = precision
        run.summary["Recall [train]"] = recall
        run.summary["F1 [train]"] = fbeta

    predict_eval = pipe.predict(X_val)

    fbeta = fbeta_score(y_val, predict_eval, beta=1, pos_label=">50K", zero_division=1)
    precision = precision_score(y_val, predict_eval, pos_label=">50K", zero_division=1)
    recall = recall_score(y_val, predict_eval, pos_label=">50K", zero_division=1)
    acc = accuracy_score(y_val, predict_eval)

    if run:   
        run.summary["Acc [validation]"] = acc
        run.summary["Precision [validation]"] = precision
        run.summary["Recall [validation]"] = recall
        run.summary["F1 [validation]"] = fbeta

    run.finish()
