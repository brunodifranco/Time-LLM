from typing import List, Tuple
from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame
import torch
from sklearn.preprocessing import MinMaxScaler
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.utils import augment_calendar_df
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer


torch.set_float32_matmul_precision("medium")

# URLs úteis
# https://www.snirh.gov.br/hidrotelemetria/serieHistorica.aspx
# https://app.powerbi.com/view?r=eyJrIjoiYWNhZDg0MDgtNWI5Ni00NjU3LTgyNDctYjgyOTE5MDFiMWM0IiwidCI6IjE1ZGNkOTA5LThkYzAtNDBlOS1hMWU1LWNlY2IwNTNjZGQxYSJ9

TOKEN = ""
HORIZON = 96
HIDDEN_LAYERS = 18
EPOCHS = 1000
BATCH_SIZE = 8
WINDOWS_BATCH_SIZE = 2
PROMPT_PREFIX = """
    The dataset contains data on Guaiba River water level in cm.
    The frequency is 15 min."
"""


class WaterLevelTrain:
    """
    A class to train a model for predicting water levels in the Guaiba River using TimeLLM.

    Attributes
    ----------
    output_path : Path
        Path to save the output data.
    variables : List
        List of variable names for the dataset.
    y_scaler : MinMaxScaler
        Scaler for the target variable.
    lag_scaler : MinMaxScaler
        Scaler for the lagged variable.
    llm_config : LlamaConfig
        Configuration for the LLM model.
    llm_model : LlamaModel
        Pre-trained LLM model.
    llm_tokenizer : LlamaTokenizer
        Tokenizer for the LLM model.
    """

    def __init__(self, output_path: Path, variables: List):
        """
        Initializes the WaterLevelTrain class with specified parameters.

        Parameters
        ----------
        input_path : Path
            Path to the input data.
        output_path : Path
            Path to save the output data.
        variables : List
            List of variable names for the dataset.
        """
        self.output_path = output_path
        self.variables = variables
        self.y_scaler = None
        self.lag_scaler = None
        self.llm_config = LlamaConfig.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_auth_token=TOKEN
        )
        self.llm_model = LlamaModel.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_auth_token=TOKEN, config=self.llm_config
        )
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_auth_token=TOKEN
        )

    def load_data(self) -> DataFrame:
        """
        Loads the data from a CSV file and preprocesses it.

        Returns
        -------
        DataFrame
            Preprocessed data.
        """
        df = pd.read_csv(
            f"elevacao_guaiba.csv"
        )  # you should run the script in the /Time-LLM/neuralforecast_timellm/guiaba folder
        df["ds"] = pd.to_datetime(df["datetime"])
        df = df.drop("datetime", axis=1)
        df = df.sort_values(by="ds", ascending=True).reset_index(drop=True)
        df = df.rename(columns={"water_level_cm": "y"})
        return df

    def feature_engineering(self, df: DataFrame) -> DataFrame:
        """
        Performs feature engineering on the dataframe.

        Parameters
        ----------
        df : DataFrame
            Input data.

        Returns
        -------
        DataFrame
            Data with engineered features.
        """
        df, _ = augment_calendar_df(df=df, freq="T")
        df["y_lag_24_hours"] = df["y"].shift(96)
        df["y_lag_24_hours"] = df["y_lag_24_hours"].fillna(df["y"])  # day lag
        df["unique_id"] = "guaiba"
        return df

    def preprocessing(self, df: DataFrame) -> DataFrame:
        """
        Scales the features using MinMaxScaler.

        Parameters
        ----------
        df : DataFrame
            Data with features to be scaled.

        Returns
        -------
        DataFrame
            Data with scaled features.
        """
        self.y_scaler = MinMaxScaler()
        self.lag_scaler = MinMaxScaler()
        df["y"] = self.y_scaler.fit_transform(df[["y"]])
        df["y_lag_24_hours"] = self.lag_scaler.fit_transform(df[["y_lag_24_hours"]])
        return df

    def train_test_split(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Splits the data into training and testing sets.

        Parameters
        ----------
        df : DataFrame
            The full dataset to be split.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Training and testing datasets.
        """
        Y_train_df = df[df.ds < df["ds"].values[-HORIZON]].copy()
        Y_test_df = df[df.ds >= df["ds"].values[-HORIZON]].reset_index(drop=True).copy()
        return Y_train_df[self.variables], Y_test_df[self.variables]

    def fit(self, Y_train_df: DataFrame) -> NeuralForecast:
        """
        Fits the TimeLLM model to the training data.

        Parameters
        ----------
        Y_train_df : DataFrame
            Training data.

        Returns
        -------
        NeuralForecast
            Fitted model.
        """
        time_llm = TimeLLM(
            h=HORIZON,
            input_size=HORIZON * 4,
            llm=self.llm_model,
            llm_config=self.llm_config,
            llm_tokenizer=self.llm_tokenizer,
            prompt_prefix=PROMPT_PREFIX,
            llm_num_hidden_layers=HIDDEN_LAYERS,
            max_steps=EPOCHS,
            batch_size=BATCH_SIZE,
            windows_batch_size=WINDOWS_BATCH_SIZE,
            num_workers_loader=15,
        )

        nf = NeuralForecast(
            models=[time_llm],
            freq="15min",
        )

        nf.fit(df=Y_train_df, val_size=HORIZON, verbose=False)
        return nf

    def predict(self, nf: NeuralForecast, Y_test_df: DataFrame) -> DataFrame:
        """
        Generates predictions using the fitted model.

        Parameters
        ----------
        nf : NeuralForecast
            Fitted model.
        Y_test_df : DataFrame
            Testing data.

        Returns
        -------
        DataFrame
            Testing data with predictions.
        """
        forecasts = nf.predict(futr_df=Y_test_df, verbose=True)
        Y_test_df["TimeLL_gpt2"] = forecasts["TimeLLM"].values
        return Y_test_df

    def inverse_scaler(
        self, Y_train_df: DataFrame, Y_test_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Applies the inverse transformation to the scaled features.

        Parameters
        ----------
        Y_train_df : DataFrame
            Training data with scaled features.
        Y_test_df : DataFrame
            Testing data with scaled features.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Data with inverse transformed features.
        """
        Y_train_df["y"] = self.y_scaler.inverse_transform(Y_train_df[["y"]])
        Y_test_df["y"] = self.y_scaler.inverse_transform(Y_test_df[["y"]])
        Y_test_df["TimeLL_gpt2"] = self.y_scaler.inverse_transform(
            Y_test_df[["TimeLL_gpt2"]]
        )

        if "y_lag_24_hours" in self.variables:
            Y_train_df["y_lag_24_hours"] = self.lag_scaler.inverse_transform(
                Y_train_df[["y_lag_24_hours"]]
            )
            Y_test_df["y_lag_24_hours"] = self.lag_scaler.inverse_transform(
                Y_test_df[["y_lag_24_hours"]]
            )

        return Y_train_df, Y_test_df

    def save_output(self, Y_train_df: DataFrame, Y_test_df: DataFrame):
        """
        Saves the training and testing data to CSV files.

        Parameters
        ----------
        Y_train_df : DataFrame
            Training data.
        Y_test_df : DataFrame
            Testing data.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        Y_train_df.to_csv(self.output_path / "y_train.csv", index=False)
        Y_test_df.to_csv(self.output_path / "y_test.csv", index=False)

    def run(self):
        """
        Runs the pipeline.

        Parameters
        ----------
        Y_train_df : DataFrame
            Training data.
        Y_test_df : DataFrame
            Testing data.
        """

        df = self.load_data()
        df = self.feature_engineering(df)
        df = self.preprocessing(df)
        Y_train_df, Y_test_df = self.train_test_split(df)
        nf = self.fit(Y_train_df)
        Y_test_df = self.predict(nf, Y_test_df)
        Y_train_df, Y_test_df = self.inverse_scaler(Y_train_df, Y_test_df)
        self.save_output(Y_train_df, Y_test_df)


if __name__ == "__main__":
    pipeline = WaterLevelTrain(
        output_path=Path("guaiaba_llama_test/"),
        variables=["ds", "unique_id", "y"],
    )
    pipeline.run()
