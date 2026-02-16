import pandas as pd
from pathlib import Path
from typing import Any
from typing import Optional
from math import ceil, log2
from Data.mappings import DECODE_MAPS_M, COLUMNS_TO_DROP_M, COLUMNS_TO_DECODE_M, COLUMNS_TO_ENCODE_M
from Data.mappings import COLUMNS_TO_DROP_D, COLUMNS_TO_DECODE_D, DECODE_MAPS_D

class Data:
    def __init__(self) -> None:
        self.raw_path: Optional[Path] = None
        self.processed_path: Optional[Path] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
    
    @staticmethod
    def extractData(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(Path(path))
        return df

    @staticmethod
    def transformData(raw_df: pd.DataFrame, parameter: str) -> pd.DataFrame:
        if parameter == 'M':
            proc_df = transform_pipeline_M(raw_df)
        else:
            proc_df = transform_pipeline_D(raw_df)
        return proc_df
    
    @staticmethod
    def loadData(path: str | Path, proc_df: pd.DataFrame):
        proc_df.to_csv(Path(path), index=False)

    def runETLforML(self, pathRaw: str | Path, pathProccessed: str | Path):
        self.raw_path = Path(pathRaw)
        self.processed_path = Path(pathProccessed)

        self.raw_df = self.extractData(self.raw_path)
        self.processed_df = self.transformData(self.raw_df, 'M')

        self.loadData(self.processed_path, self.processed_df)
    def runETLforDashboard(self, pathRaw: str | Path, pathProccessed: str | Path):
        self.raw_path = Path(pathRaw)
        self.processed_path = Path(pathProccessed)

        self.raw_df = self.extractData(self.raw_path)
        self.processed_df = self.transformData(self.raw_df, 'D')

        self.loadData(self.processed_path, self.processed_df)   


def dropColumns(df: pd.DataFrame, cl : list[str]) -> pd.DataFrame:
    missing = [c for c in cl if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found for one-hot: {missing}")
    return df.drop(columns=cl)

       
def encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found for one-hot: {missing}")
    out = df.copy()
    for col in cols:
        s = out[col].astype("object").where(out[col].notna(), "__MISSING__")
        codes, uniques = pd.factorize(s, sort=True)

        k = len(uniques)
        n_bits = max(1, ceil(log2(k)))

        for bit in range(n_bits):
            out[f"{col}_b{bit}"] = ((codes >> bit) & 1).astype("int8")

        out.drop(columns=[col], inplace=True)

    return out  

def decode(df: pd.DataFrame, cl: list[str], DECODE_MAPS: dict[str, dict[Any, Any]]) -> pd.DataFrame:
    out = df.copy()

    for col in cl:
        if col not in out.columns:
            raise KeyError(f"Column not found for decode: {col}")
        if col not in DECODE_MAPS:
            raise KeyError(f"No mapping defined in DECODE_MAPS for: {col}")

        mapping = DECODE_MAPS[col]

        if col == 'Age' or col == 'Education' or col == 'Gender' or col == 'Country':
            rounded = out[col].round(5)
            out[col] = rounded.map(mapping)
        else:
            out[col] = out[col].map(mapping)

        if out[col].isna().any():
            bad = df.loc[out[col].isna(), col].unique()[:10]
            raise ValueError(f"Unmapped codes in '{col}' (sample): {bad}")

        out[col] = out[col].astype("category")

    return out

def transform_pipeline_M(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = dropColumns(out, COLUMNS_TO_DROP_M)
    
    out = decode(out, COLUMNS_TO_DECODE_M, DECODE_MAPS_M)

    out = encode(out, COLUMNS_TO_ENCODE_M)

    return out

def transform_pipeline_D(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = dropColumns(out, COLUMNS_TO_DROP_D)
    
    out = decode(out, COLUMNS_TO_DECODE_D, DECODE_MAPS_D)
    
    return out
