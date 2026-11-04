import polars as pl
import pandas as pd
from pathlib import Path
import os

DATA_PATH_RAW = Path("./../data") / "raw"
RESULT_PATH = Path("./../data") / "results"
PROCCESED_PATH = Path("./../data") / "processed"
EXTRACTED_PATH = Path("./../data") / "raw" / "extracted"

def load_data(path: Path, dtype={}):
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path).to_pandas()
    elif ext == ".csv":
        return pd.read_csv(path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
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

def split_train_set(d, split_date, train_samples=2_000_000, validation_samples=400_000, target="unit_sales"):
    train = d[(d["date"] < split_date)].sample(n=train_samples, random_state=0)
    valid = d[d["date"] >= split_date].sample(n=validation_samples, random_state=0)
    features = train.columns.drop([target])
    x_train, y_train = train[features], train[target]
    x_valid, y_valid = valid[features], valid[target]
    return (x_train, y_train, x_valid, y_valid)


def write_result(data: pd.DataFrame, filename : str):
    # write to csv
    os.makedirs(RESULT_PATH, exist_ok=True)
    data.to_csv(RESULT_PATH / filename, index=False)