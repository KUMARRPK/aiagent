import re
from pathlib import Path

import duckdb
import pandas as pd

from core.config import GOLD_DIR
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


def _fix_table_names(sql: str, real_table_names: list[str]) -> str:
    for real_name in real_table_names:
        stripped = re.sub(r"_[0-9a-f]{8}$", "", real_name)
        if stripped == real_name:
            continue
        sql = re.sub(rf"\b{re.escape(stripped)}\b", real_name, sql)
    return sql


def run(
    silver_paths: list[str], sttm_path: str,
    business_intent: str, run_id: str
) -> list[str]:
    audit = AuditLogger(run_id)
    sttm_df = pd.read_csv(sttm_path)
    conn = duckdb.connect()
    table_registry: dict[str, str] = {}

    schema_lines = []
    for fp in silver_paths:
        clean_name = Path(fp).stem
        conn.execute(
            f"CREATE OR REPLACE TABLE {clean_name} AS "
            f"SELECT * FROM read_parquet('{fp}')"
        )
        table_registry[clean_name] = fp
        row_count = conn.execute(f"SELECT COUNT(*) FROM {clean_name}").fetchone()[0]
        columns = conn.execute(f"DESCRIBE {clean_name}").df()["column_name"].tolist()
        schema_lines.append(f"Table {clean_name}: {row_count} rows, columns={columns}")

    prompt = (
        f"Business intent: {business_intent}\n\n"
        f"Table schemas:\n" + "\n".join(schema_lines) + "\n\n"
        f"Gold STTM rules:\n{sttm_df.to_string(index=False)}\n\n"
        "Instructions:\n"
        "- Parse join rules 'join_left:A:B:key' as 'A LEFT JOIN B ON A.key = B.key'\n"
        "- NEVER join on a pk_*_silver_id column\n"
        "- Check exact column lists for BOTH tables before writing any join\n"
        "- Any *_date column is stored as VARCHAR -- always wrap in CAST(col AS DATE)\n"
        "- DuckDB has NO DATEADD() -- use date - INTERVAL 'N months' instead\n"
        "- For time_filter anchored on MAX(date), use "
        "(SELECT MAX(date_col::DATE) FROM fact_table) NOT CURRENT_DATE\n"
        "- Parse rank_limit as ORDER BY + LIMIT clause\n"
        "- Generate ONE DuckDB SELECT with JOINs, GROUP BY, aggregates\n"
        "- Do NOT add pk_gold_id (added in Python)\n"
        "- Order by the main aggregate DESC\n"
        "Respond with a ```sql fenced code block containing ONLY the SQL."
    )

    df = None
    sql_used = None
    sql_error = None

    row_counts = {
        name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in table_registry
    }
    fallback_table = max(row_counts, key=row_counts.get)

    llm_text = ""
    try:
        llm_text = call_llm(prompt, max_tokens=3500, reasoning_effort="medium")
        sql = _extract_sql(llm_text)
        if sql:
            sql = _fix_table_names(sql, list(table_registry.keys()))
            df = conn.execute(sql).df()
            sql_used = sql
        else:
            sql_error = "No SQL could be extracted from the LLM response."
    except Exception as e:
        sql_error = str(e)
        df = None

    if df is None:
        debug_path = GOLD_DIR / f"gold_sql_failure_{run_id[:8]}.txt"
        debug_path.write_text(
            f"ERROR: {sql_error}\n\n--- RAW LLM RESPONSE ---\n{llm_text}\n",
            encoding="utf-8",
        )
        audit.log("gold_agent", "sql_generation_failed",
                  error=sql_error, debug_file=str(debug_path))
        df = pd.read_parquet(table_registry[fallback_table])
        sql_used = f"FALLBACK: pd.read_parquet({fallback_table})"

    elif len(df) == 0:
        debug_path = GOLD_DIR / f"gold_sql_zero_rows_{run_id[:8]}.txt"
        debug_path.write_text(
            f"SQL executed successfully but returned 0 rows.\n\n"
            f"--- SQL EXECUTED ---\n{sql_used}\n",
            encoding="utf-8",
        )
        audit.log("gold_agent", "sql_returned_zero_rows",
                  sql=sql_used, debug_file=str(debug_path))

    df.insert(0, "pk_gold_id", range(1, len(df) + 1))
    out_path = GOLD_DIR / f"sales_analytics_gold_{run_id[:8]}.parquet"
    df = sanitize_for_parquet(df)
    df.to_parquet(out_path, index=False)

    audit.log("gold_agent", "gold_table_generated",
              output_path=str(out_path), row_count=len(df), sql=sql_used)

    return [str(out_path)]