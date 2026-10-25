"""Download every insider open-market purchase (SEC Form 4, code "P") since 2016.

    python -m trader.insider_data

Source: the SEC's quarterly "Insider Transactions Data Sets" (public, free). Keeps purchases of
common stock by officers and directors (and 10% owners, flagged), with the filing date (when the
public could first see it). Saved to research/insider_buys.csv.gz for the lab (trader/lab.py).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
import zipfile
from datetime import date

import pandas as pd
import requests

log = logging.getLogger("insider")

URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
UA = "larp-trading research {contact}"   # the SEC refuses requests without a contact email


def quarters(start: int = 2016) -> list[str]:
    today = date.today()
    out = []
    for y in range(start, today.year + 1):
        for q in range(1, 5):
            if (y, q) < (today.year, (today.month - 1) // 3 + 1):
                out.append(f"{y}q{q}")
    return out


def _tsv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    member = next(n for n in z.namelist() if n.upper().endswith(name))
    df = pd.read_csv(z.open(member), sep="\t", dtype=str, quoting=3, on_bad_lines="skip", low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df


def purchases(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    sub = _tsv(z, "SUBMISSION.TSV")[["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"]]
    own = _tsv(z, "REPORTINGOWNER.TSV")
    own = own[[c for c in ("ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE") if c in own]]
    own = own.groupby("ACCESSION_NUMBER").first().reset_index()
    tr = _tsv(z, "NONDERIV_TRANS.TSV")
    tr = tr[(tr["TRANS_CODE"] == "P") & (tr.get("TRANS_ACQUIRED_DISP_CD", "A") == "A")]
    tr = tr[["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]]
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    df = df[df["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    shares = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    out = pd.DataFrame({
        "symbol": df["ISSUERTRADINGSYMBOL"].str.upper().str.strip(),
        "filed": pd.to_datetime(df["FILING_DATE"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d"),
        "traded": pd.to_datetime(df["TRANS_DATE"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d"),
        "owner": df.get("RPTOWNERCIK"),
        "relationship": df.get("RPTOWNER_RELATIONSHIP"),
        "title": df.get("RPTOWNER_TITLE"),
        "value": (shares * price).round(0),
    })
    return out.dropna(subset=["symbol", "filed", "value"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--out", default="research/insider_buys.csv.gz")
    ap.add_argument("--contact", required=True, help="contact email the SEC asks for (not stored anywhere)")
    args = ap.parse_args()
    frames = []
    for q in quarters(args.start):
        for attempt in range(4):
            r = requests.get(URL.format(q=q), headers={"User-Agent": UA.format(contact=args.contact)}, timeout=120)
            if r.status_code == 200:
                break
            log.warning("%s: HTTP %s, retrying", q, r.status_code)
            time.sleep(5 * (attempt + 1))
        if r.status_code != 200:
            log.warning("%s: skipped (HTTP %s)", q, r.status_code)
            continue
        try:
            p = purchases(r.content)
        except Exception as e:
            log.warning("%s: could not parse (%s)", q, e)
            continue
        frames.append(p)
        log.info("%s: %d purchases", q, len(p))
        time.sleep(1)   # the SEC asks for at most 10 requests a second; be gentler than that
    if not frames:
        raise SystemExit("no quarter could be downloaded")
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    log.info("saved %d purchases", len(df))


if __name__ == "__main__":
    main()
