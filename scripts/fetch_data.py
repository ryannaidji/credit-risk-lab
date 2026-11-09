"""Rebuild the committed snapshots in data/ from their public sources. Needs xlrd and lxml."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "Mozilla/5.0"}
UCI = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
FED = "https://www.federalreserve.gov/releases/chargeoff/delallsa.htm"


def credit() -> None:
    raw = requests.get(UCI, headers=UA, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        df = pd.read_excel(io.BytesIO(z.read(name)), header=1)
    df = df.rename(columns={"default payment next month": "default"})
    df.to_csv(DATA / "credit_default.csv", index=False)


def delinquency() -> None:
    html = requests.get(FED, headers=UA, timeout=120).text
    table = next(t for t in pd.read_html(io.StringIO(html)) if t.shape[0] > 100)
    out = pd.DataFrame({"quarter": table.iloc[:, 0], "rate": table.iloc[:, 6]})
    out = out[out["rate"] != "n.a."].copy()
    out["rate"] = out["rate"].astype(float)
    out = out.iloc[::-1].reset_index(drop=True)
    out.to_csv(DATA / "cc_delinquency_sa.csv", index=False)


if __name__ == "__main__":
    credit()
    delinquency()
