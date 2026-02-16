from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


class Model:
    """
    ANN pentru regresie: prezice coloana `SS` din dataset.
    Obiectul se construiește doar cu `data` (DataFrame sau path către CSV).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str = "SS",
        id_col: str = "ID",
        seed: int = 42,
    ) -> None:
        self.seed = seed
        set_seed(seed)
        self.df = data

        self.target = target
        self.id_col = id_col

        self.data = data

        self.splits: Optional[SplitData] = None
        self.model: Optional[tf.keras.Model] = None
        self.metrics: Optional[list] = None
        self.history: Optional[tf.keras.callbacks.History] = None

    def prepare_data(
        self,
        test_size: float = 0.15,
        val_size_from_remaining: float = 0.1765,  #70/15/15
        random_state: Optional[int] = None,
    ) -> SplitData:
        
        rs = self.seed if random_state is None else random_state

        drop_cols = [self.target]
        if self.id_col in self.df.columns:
            drop_cols.append(self.id_col)

        feature_df = self.df.drop(columns=drop_cols)
        y = self.df[self.target].to_numpy(dtype=np.float32)
        X = feature_df.to_numpy(dtype=np.float32)
        feature_names = list(feature_df.columns)

        # 70/15/15
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_from_remaining, random_state=rs
        )

        self.splits = SplitData(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
        )
        return self.splits

    def build_model(
        self,
        l2_strength: float = 1e-4,
        dropout: float = 0.15,
        hidden1: int = 128,
        hidden2: int = 64,
        learning_rate: float = 1e-3,
    ) -> tf.keras.Model:
        
        if self.splits is None:
            self.prepare_data()

        assert self.splits is not None
        input_dim = self.splits.X_train.shape[1]
        reg = tf.keras.regularizers.l2(l2_strength)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden1, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden2, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="linear"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=[
                    tf.keras.metrics.MeanAbsoluteError(name="mae"),
                    tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                ],
        )

        self.model = model
        return model

    def train(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: int = 1,
        callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
    ) -> tf.keras.callbacks.History:
        if self.splits is None:
            self.prepare_data()
        if self.model is None:
            self.build_model()

        assert self.splits is not None
        assert self.model is not None

        cb = callbacks if callbacks is not None else [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            )
        ]

        self.history = self.model.fit(
            self.splits.X_train, self.splits.y_train,
            validation_data=(self.splits.X_val, self.splits.y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=cb,
            verbose=verbose
        )
        return self.history

    def evaluate(self) -> Dict[str, float]:
        if self.splits is None or self.model is None:
            raise RuntimeError("Apelează prepare_data() și train() înainte de evaluate().")

        values = self.model.evaluate(self.splits.X_test, self.splits.y_test, verbose=0)
        return dict(zip(self.model.metrics_names, map(float, values)))

    def predict(self, n: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.splits is None or self.model is None:
            raise RuntimeError("Apelează prepare_data() și train() înainte de predict().")

        y_true = self.splits.y_test[:n]
        y_pred = self.model.predict(self.splits.X_test[:n], verbose=0).reshape(-1)
        abs_err = np.abs(y_pred - y_true)
        return y_true, y_pred, abs_err

    def save_model(self, path: str | Path = "ss_model.keras") -> None:
        if self.model is None:
            raise RuntimeError("Nu există model. Apelează build_model()/train() înainte de save_model().")
        self.model.save(str(path))

    def load_model(self, path: str | Path) -> tf.keras.Model:
        self.model = tf.keras.models.load_model(str(path))
        return self.model

    def run_model(  
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        save_path: str | Path = "ss_model.keras",
        verbose: int = 1,
    ) -> None:

        self.df = data
        self.splits = None
        self.model = None
        self.history = None

        self.prepare_data()

        self.build_model()

        self.train(epochs=epochs, batch_size=batch_size, verbose=verbose)

        self.evaluate()

        self.save_model(save_path)


# ------------------- Exemplu de utilizare -------------------
if __name__ == "__main__":
    df = pd.read_csv(r"C:\CORUNA\DataVi\Project\DVA_Final_Project\Dataset\Processed\drug_consumption_processed_ml.csv")

    trainer = Model(df)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()

    metrics = trainer.evaluate()
    print("\nTEST metrics:", metrics)

    y_true, y_pred, abs_err = trainer.predict(n=10)
    print("\nSS reale (primele 10):   ", np.round(y_true, 4))
    print("SS prezise (primele 10): ", np.round(y_pred, 4))
    print("Abs err (primele 10):    ", np.round(abs_err, 4))

    trainer.save_model("ss_model.keras")
