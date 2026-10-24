import io
import zipfile

from trader.insider_data import purchases, quarters


def _zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERTRADINGSYMBOL\n"
                   "a1\t05-MAR-2024\t4\tacme\na2\t06-MAR-2024\t4\tACME\na3\t06-MAR-2024\t3\tACME\n")
        z.writestr("REPORTINGOWNER.tsv", "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
                   "a1\t111\tOfficer\tCEO\na2\t222\tDirector\t\na3\t333\tDirector\t\n")
        z.writestr("NONDERIV_TRANS.tsv", "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
                   "a1\t04-MAR-2024\tP\t1000\t50\tA\na2\t05-MAR-2024\tS\t500\t50\tD\na3\t05-MAR-2024\tP\t10\t50\tA\n")
    return b.getvalue()


def test_keeps_only_open_market_purchases_on_form_4():
    df = purchases(_zip())
    assert df.to_dict("records") == [{"symbol": "ACME", "filed": "2024-03-05", "traded": "2024-03-04", "owner": "111",
                                      "relationship": "Officer", "title": "CEO", "value": 50000.0}]


def test_quarters_stop_before_the_current_one():
    q = quarters(2016)
    assert q[0] == "2016q1" and len(q) >= 40
