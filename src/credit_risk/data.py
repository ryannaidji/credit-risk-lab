"""Load and clean the public datasets; define the one split shared by Python and R."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SEED = 42

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
AMOUNTS = ["LIMIT_BAL"] + BILL + PAY_AMT
NUMERIC = ["LIMIT_BAL", "AGE"] + PAY + BILL + PAY_AMT
FEATURES = CATEGORICAL + NUMERIC
TARGET = "default"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Merge undocumented category codes into the other group (mirrored in r/glm_benchmark.R)."""
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].isin([1, 2, 3]), 4)
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].isin([1, 2]), 3)
    return out


def load_credit() -> pd.DataFrame:
    return clean(pd.read_csv(DATA / "credit_default.csv"))


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/30 split by customer ID."""
    train, test = train_test_split(df["ID"], test_size=0.3, random_state=SEED, stratify=df[TARGET])
    return pd.concat(
        [pd.DataFrame({"ID": train, "set": "train"}), pd.DataFrame({"ID": test, "set": "holdout"})]
    ).sort_values("ID")


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_credit()
    split = pd.read_csv(DATA / "split.csv")
    merged = df.merge(split, on="ID", validate="one_to_one")
    return merged[merged["set"] == "train"].copy(), merged[merged["set"] == "holdout"].copy()


def load_delinquency() -> pd.Series:
    d = pd.read_csv(DATA / "cc_delinquency_sa.csv")
    return pd.Series(d["rate"].values, index=d["quarter"], name="rate")


if __name__ == "__main__":
    make_split(load_credit()).to_csv(DATA / "split.csv", index=False)
