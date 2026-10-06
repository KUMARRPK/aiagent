import re
from pathlib import Path

import duckdb
import pandas as pd

from core.config import SILVER_DIR
from core.audit import AuditLogger
from core.llm import call_llm
from core.parquet_safety import sanitize_for_parquet


def _extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _base_name(bronze_table_name: str, run_id: str) -> str:
    return bronze_table_name.replace(f"_bronze_{run_id[:8]}", "")


def _fix_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUG FIX: When silver falls back to pd.read_parquet + drop_duplicates,
    date standardization never runs. This ensures all date columns are
    properly parsed regardless of mixed formats (YYYY-MM-DD, MM/DD/YYYY,
    MM-DD-YYYY) so the Gold time filter never excludes NULL date rows.
    """
    for col in df.columns:
        if "date" not in col.lower():
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str).str.split(" ").str[0]
            continue
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(
                    df[col], format="mixed", dayfirst=False, errors="coerce"
                )
                df[col] = parsed.dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return df


def run(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    audit = AuditLogger(run_id)
    sttm_df = pd.read_csv(sttm_path)
    written_paths: list[str] = []
    conn = duckdb.connect()

    for fp in bronze_paths:
        table_alias = Path(fp).stem
        base_name = _base_name(table_alias, run_id)

        conn.execute(
            f"CREATE OR REPLACE TABLE {table_alias} AS "
            f"SELECT * FROM read_parquet('{fp}')"
        )

        df_preview = conn.execute(f"SELECT * FROM {table_alias} LIMIT 200").df()
        categorical_distincts = {}
        for col in df_preview.select_dtypes(include="object").columns:
            categorical_distincts[col] = df_preview[col].dropna().unique()[:10].tolist()

        table_rules = sttm_df[sttm_df["source_table"] == base_name]

        prompt = (
            f"Table (DuckDB alias): {table_alias}\n"
            f"Schema: {list(df_preview.columns)}\n"
            f"Categorical distinct value samples: {categorical_distincts}\n\n"
            f"STTM rules for this table:\n{table_rules.to_string(index=False)}\n\n"
            "Generate ONE DuckDB SELECT SQL against this table implementing:\n"
            "- standardize_date: COALESCE(TRY_STRPTIME(col,'%Y-%m-%d'), "
            "TRY_STRPTIME(col,'%m/%d/%Y'), "
            "TRY_STRPTIME(col,'%m-%d-%Y'))::VARCHAR "
            "to handle ALL mixed date formats including MM-DD-YYYY\n"
            "- normalize_category: CASE WHEN with patterns\n"
            "- null_handler drop_nulls: WHERE col IS NOT NULL\n"
            "- null_handler fill_median: COALESCE(col, median subquery)\n"
            "- dedup: SELECT DISTINCT\n"
            "- surrogate_key: IGNORE this rule entirely\n"
            "- passthrough: SELECT col AS col unchanged\n"
            f"Select FROM {table_alias}.\n"
            "Respond with a ```sql fenced code block containing ONLY the SQL."
        )

        df = None
        sql_used = None
        try:
            llm_text = call_llm(prompt, max_tokens=2500)
            sql = _extract_sql(llm_text)
            if sql:
                df = conn.execute(sql).df()
                sql_used = sql
        except Exception:
            df = None

        if df is None:
            df = pd.read_parquet(fp).drop_duplicates()
            sql_used = "FALLBACK: pd.read_parquet + drop_duplicates + date_fix"

        # FIX: Always run date fix regardless of LLM or fallback path
        df = _fix_date_columns(df)

        df.insert(0, f"pk_{base_name}_silver_id", range(1, len(df) + 1))
        df = sanitize_for_parquet(df)

        out_path = SILVER_DIR / f"{base_name}_silver_{run_id[:8]}.parquet"
        df.to_parquet(out_path, index=False)
        written_paths.append(str(out_path))

        date_nulls = sum(
            df[c].isna().sum()
            for c in df.columns if "date" in c.lower()
        )

        audit.log(
            "silver_agent", "table_processed",
            table=base_name, output_path=str(out_path),
            row_count=len(df), sql=sql_used,
            date_nulls_after_fix=int(date_nulls),
        )

    return written_paths