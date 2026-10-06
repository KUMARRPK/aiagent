import re
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from core.config import BRONZE_DIR
from core.audit import AuditLogger
from core.llm import call_llm
from core.parquet_safety import sanitize_for_parquet

# Date formats present in raw data — used for fallback fix
_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"]


def _extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _fix_dates_from_raw(df: pd.DataFrame, fp: str) -> pd.DataFrame:
    """
    BUG FIX: TRY_CAST(date AS DATE) silently fails for MM/DD/YYYY and
    MM-DD-YYYY formats, turning 32% of rows to NULL.
    This function repairs those NULLs after bronze load.
    """
    try:
        raw = pd.read_csv(fp, usecols=lambda c: "date" in c.lower())
        for col in raw.columns:
            if col in df.columns:
                parsed = pd.to_datetime(
                    raw[col], format="mixed", dayfirst=False, errors="coerce"
                )
                null_mask = df[col].isna()
                df.loc[null_mask, col] = parsed[null_mask].values
    except Exception:
        pass
    return df


def run(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    audit = AuditLogger(run_id)
    sttm_df = pd.read_csv(sttm_path)
    written_paths: list[str] = []

    for fp in input_files:
        table_name = Path(fp).stem
        table_rules = sttm_df[sttm_df["source_table"] == table_name]
        sample_df = pd.read_csv(fp, nrows=3)

        prompt = (
            f"CSV file path: {fp}\n"
            f"Table: {table_name}\n\n"
            f"STTM rules for this table:\n{table_rules.to_string(index=False)}\n\n"
            f"Column type sample (first 3 rows):\n{sample_df.to_string(index=False)}\n\n"
            "Generate ONE DuckDB SELECT SQL using read_csv_auto() with "
            "all_varchar=true. Use TRY_CAST() for type_cast rules. Map: "
            "float->DOUBLE, int->INTEGER, str->VARCHAR. "
            "IMPORTANT: For any date/datetime column, do NOT use TRY_CAST(col AS DATE). "
            "Instead use: COALESCE(TRY_STRPTIME(col,'%Y-%m-%d'), "
            "TRY_STRPTIME(col,'%m/%d/%Y'), "
            "TRY_STRPTIME(col,'%m-%d-%Y'))::VARCHAR AS col "
            "to handle mixed date formats without data loss.\n"
            "Do NOT add metadata columns in SQL (added afterward in Python).\n"
            "Respond with a ```sql fenced code block containing ONLY the SQL."
        )

        df = None
        sql_used = None
        try:
            llm_text = call_llm(prompt, max_tokens=1500)
            sql = _extract_sql(llm_text)
            if sql:
                conn = duckdb.connect()
                df = conn.execute(sql).df()
                sql_used = sql
        except Exception:
            df = None

        if df is None:
            df = pd.read_csv(fp)
            sql_used = "FALLBACK: pd.read_csv passthrough"

        # FIX: Repair any NULL dates caused by TRY_CAST failure
        df = _fix_dates_from_raw(df, fp)

        df["_load_timestamp"] = datetime.utcnow().isoformat() + "Z"
        df["_source_file"] = str(fp)

        out_path = BRONZE_DIR / f"{table_name}_bronze_{run_id[:8]}.parquet"
        df = sanitize_for_parquet(df)
        df.to_parquet(out_path, index=False)
        written_paths.append(str(out_path))

        date_nulls = sum(
            df[c].isna().sum()
            for c in df.columns if "date" in c.lower()
        )

        audit.log(
            "bronze_agent", "table_processed",
            table=table_name, output_path=str(out_path),
            row_count=len(df), sql=sql_used,
            date_nulls_after_fix=int(date_nulls),
        )

    return written_paths