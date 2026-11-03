import polars as pl
import pandas as pd
from pathlib import Path

DATA_PATH_RAW = Path("./../data") / "raw"
RESULT_PATH = Path("./../data") / "results"
PROCCESED_PATH = Path("./../data") / "processed"

def load_data(path: Path):
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path).to_pandas()
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
def clean_features(d : pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["onpromotion"] = d["onpromotion"].fillna(False)
    return d

def add_date_features(d : pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # adds weekday and month feature
    d["date"] = pd.to_datetime(d["date"])
    d["weekday_num"] = d["date"].dt.day_of_week
    d["month_num"] = d["date"].dt.month
    return d.drop(columns="date")

def add_zero_sales(data: pd.DataFrame) -> pd.DataFrame:
    data["unit_sales"] = data["unit_sales"].clip(lower=0)

    # adds 0 sales for missing item sales
    sales = data.groupby(["date", "store_nbr", "item_nbr"])["unit_sales"].sum()

    all_dates = sales.index.get_level_values("date").unique()
    all_items = sales.index.get_level_values("item_nbr").unique()
    all_shops = sales.index.get_level_values("store_nbr").unique()

    full_index = pd.MultiIndex.from_product([all_dates, all_shops, all_items], names=["date", "store_nbr", "item_nbr"])

    data_zeros = sales.reindex(full_index, fill_value=0).reset_index()

    return data_zeros.merge(data.drop(columns="unit_sales"), on=["date", "item_nbr", "store_nbr"], how="left")