"""
FIXER AGENT — Gemini version
Generates fix code for failed checks AND always adds _dq_flag columns.
"""
import re
import pandas as pd
import numpy as np
from agents.gemini_client import generate

SYSTEM = """You are a pandas data-cleaning code writer.
Write a Python function called apply_fix(df) that:
- Accepts a pandas DataFrame df
- Fixes the specific issue described
- Returns the fixed DataFrame
- Is safe — does not destroy data, only cleans/repairs
Return ONLY raw Python code. No markdown. No explanation.
Start with: def apply_fix(df):"""


class FixerAgent:
    def run(self, df: pd.DataFrame, failures: list, emit=None):
        fixed_df    = df.copy()
        fixes_applied = 0

        for f in failures:
            col        = f.get("column_name", "")
            rule_label = f.get("rule_label", "")
            if not col or col not in fixed_df.columns:
                continue

            if emit:
                emit("Fixer", f"Generating fix for: {rule_label} on '{col}'", "info")

            # ── Step 1: Always add a _dq_flag column for this failure ──────────
            flag_col = f"{col}_dq_flag"
            if flag_col not in fixed_df.columns:
                fixed_df[flag_col] = "PASS"

            # Mark failing rows using heuristic mask
            mask = self._get_fail_mask(fixed_df[col], rule_label)
            fixed_df.loc[mask, flag_col] = f"FAIL - {rule_label}"
            f["fix_applied"] = True
            fixes_applied += 1
            if emit:
                fail_count = int(mask.sum())
                emit("Fixer", f"✓ Flag column '{flag_col}' added ({fail_count} rows marked FAIL)", "success")

            # ── Step 2: Try to auto-fix the value in place ────────────────────
            try:
                code   = self._gen_fix(fixed_df, f)
                ns     = {}
                exec(compile(code, "<dq_fix>", "exec"),
                     {"pd": pd, "np": np, "__builtins__": __builtins__}, ns)
                fn = ns.get("apply_fix")
                if fn:
                    fixed_df = fn(fixed_df)
                    if emit:
                        emit("Fixer", f"✓ Value fix also applied for {col}", "success")
            except Exception as e:
                if emit:
                    emit("Fixer", f"⚠ Value fix skipped for {col}: {e}", "warn")

        return fixed_df, fixes_applied

    # ── Heuristic mask: identify which rows fail a given rule ──────────────────
    def _get_fail_mask(self, s: pd.Series, rule_label: str) -> pd.Series:
        label = rule_label.lower()
        n     = len(s)
        false_series = pd.Series([False] * n, index=s.index)

        try:
            # Null / missing
            if any(k in label for k in ["null","missing","not null","required","mandatory","empty"]):
                return s.isna() | (s.astype(str).str.strip() == "")

            # Negative values
            if any(k in label for k in ["negative","non-negative","positive",">= 0","> 0"]):
                num = pd.to_numeric(s, errors="coerce")
                return num < 0

            # Email format
            if any(k in label for k in ["email","e-mail","mail format"]):
                pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
                return ~s.astype(str).str.match(pattern) & s.notna()

            # Phone format
            if any(k in label for k in ["phone","mobile","contact number"]):
                cleaned = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
                return ~cleaned.str.match(r"^\d{7,15}$") & s.notna()

            # Unique / duplicate
            if any(k in label for k in ["unique","duplicate","distinct"]):
                return s.duplicated(keep=False)

            # Date format / future date
            if any(k in label for k in ["date","future","past","datetime"]):
                parsed = pd.to_datetime(s, errors="coerce")
                if "future" in label:
                    return parsed > pd.Timestamp.now()
                if "past" in label or "not future" in label:
                    return parsed > pd.Timestamp.now()
                return parsed.isna() & s.notna()

            # Range check — parse min/max from label
            range_min = re.search(r"(?:>=|≥|min|minimum|at least|>)\s*([0-9]+(?:\.[0-9]+)?)", label)
            range_max = re.search(r"(?:<=|≤|max|maximum|at most|<)\s*([0-9]+(?:\.[0-9]+)?)", label)
            if range_min or range_max:
                num  = pd.to_numeric(s, errors="coerce")
                fail = pd.Series([False] * n, index=s.index)
                if range_min:
                    fail = fail | (num < float(range_min.group(1)))
                if range_max:
                    fail = fail | (num > float(range_max.group(1)))
                return fail

            # Allowable values
            allowed = re.search(
                r"(?:one of|in\s*\[|allowed|valid values?|must be)\s*[:\[]?\s*([\w,\s'\"]+)", label)
            if allowed:
                vals = {v.strip().strip("'\"").lower()
                        for v in allowed.group(1).split(",") if v.strip()}
                if vals:
                    return ~s.astype(str).str.lower().isin(vals) & s.notna()

        except Exception:
            pass

        return false_series

    def _gen_fix(self, df: pd.DataFrame, failure: dict) -> str:
        prompt = (
            f"DataFrame columns: {df.columns.tolist()}\n"
            f"Dtypes: { {c: str(t) for c, t in df.dtypes.items()} }\n\n"
            f"Generate apply_fix(df) to fix:\n"
            f"  Column: {failure['column_name']}\n"
            f"  Issue: {failure['rule_label']}\n"
            f"  Details: {failure.get('details', '')[:200]}\n"
            f"  Affected rows: {failure.get('affected', 0)}"
        )
        code = generate(prompt, system=SYSTEM)
        code = re.sub(r"```(?:python)?", "", code).replace("```", "").strip()
        return code
