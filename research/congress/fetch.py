"""Every stock trade members of the House disclosed (Periodic Transaction Reports), one year per run.

    python research/congress/fetch.py 2024 --out research/congress/trades_2024.csv.gz

Reads the House Clerk's yearly index, downloads each PTR PDF slowly, turns it into text with `pdftotext -layout`
and into rows with the parser from github.com/Jakeschincariol/trader-skill (ptr.py, MIT). Scanned (handwritten)
reports give no rows and are counted. Columns: member, doc_id, filed, owner, ticker, type (P/S/S (partial)/E),
date (trade), amount, amount_low, amount_high, asset_type.
"""
import argparse, csv, gzip, logging, os, sys, tempfile, time

sys.path.insert(0, os.path.dirname(__file__))
import ptr  # noqa: E402

log = logging.getLogger("congress")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pause", type=float, default=0.7)
    a = ap.parse_args()
    filings = [f for f in ptr.load_index(a.year) if f["type"] == "P"]
    log.info("%d: %d transaction reports", a.year, len(filings))
    rows, empty, failed = [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(filings):
            time.sleep(a.pause)
            try:
                data = ptr.download(f["url"])
            except ptr.PtrError as e:
                failed += 1
                log.warning("%s: %s", f["doc_id"], e)
                continue
            path = os.path.join(tmp, "p.pdf")
            open(path, "wb").write(data)
            text = ptr.pdf_to_text(path) or ""
            got = ptr.parse_ptr_text(text, f["filed"])["rows"] if text.strip() else []
            if not got:
                empty += 1
            for r in got:
                rows.append({"member": f["name"], "doc_id": f["doc_id"], "filed": f["filed"], "owner": r.get("owner"),
                             "ticker": r.get("ticker"), "type": r.get("type"), "date": r.get("date"),
                             "amount": r.get("amount"), "amount_low": r.get("amount_low"),
                             "amount_high": r.get("amount_high"), "asset_type": r.get("asset_type")})
            if k % 200 == 0:
                log.info("%d/%d reports, %d trades, %d unreadable, %d failed", k + 1, len(filings), len(rows), empty, failed)
    with gzip.open(a.out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["member", "doc_id", "filed", "owner", "ticker", "type", "date", "amount",
                                           "amount_low", "amount_high", "asset_type"])
        w.writeheader()
        w.writerows(rows)
    log.info("%d done: %d trades from %d reports (%d unreadable, %d failed)", a.year, len(rows), len(filings), empty, failed)


if __name__ == "__main__":
    main()
