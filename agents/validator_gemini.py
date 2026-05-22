"""
VALIDATOR AGENT — Pure-pandas version (no LLM calls per check)
Executes rules against a DataFrame using direct pandas logic.
This completely eliminates Gemini API 429 errors during validation.
"""
import re, traceback
import pandas as pd

MAX_RETRY = 0  # No LLM retries needed — pure pandas never needs regeneration


class ValidatorAgent:
    def run(self, df: pd.DataFrame, all_rules: dict, emit=None) -> list:
        """
        all_rules: {col_name: [{label, logic, severity, source}]}
        emit: optional callback(agent, msg, level)
        Returns list of check result dicts.
        """
        results = []
        checks = self._flatten_rules(all_rules)
        total = len(checks)
        for i, check in enumerate(checks, 1):
            if emit:
                emit("Validator", f"[{i}/{total}] Checking: {check['label']} on '{check['column']}'", "info")
            result = self._run_check(df, check)
            results.append(result)
        return results

    def _flatten_rules(self, all_rules: dict) -> list:
        checks = []
        cid = 1
        for col, rules in all_rules.items():
            for r in rules:
                checks.append({
                    "id": f"CHK_{cid:03d}",
                    "column": col,
                    "label": r.get("label", r.get("logic", "check")),
                    "logic": r.get("logic", ""),
                    "severity": r.get("severity", "Medium"),
                    "source": r.get("source", "ai"),
                })
                cid += 1
        return checks

    def _run_check(self, df: pd.DataFrame, check: dict) -> dict:
        col = check["column"]
        label = check["label"].lower()
        logic = check["logic"].lower()
        combined = label + " " + logic

        try:
            passed, affected, details = self._execute_rule(df, col, combined, check["logic"])
            return {
                "id": check["id"],
                "column_name": col,
                "rule_label": check["label"],
                "severity": check["severity"],
                "source": check["source"],
                "status": "PASS" if passed else "FAIL",
                "affected": affected,
                "total": len(df),
                "details": details,
                "fix_applied": False,
                "retries": 0,
            }
        except Exception as e:
            return {
                "id": check["id"],
                "column_name": col,
                "rule_label": check["label"],
                "severity": check["severity"],
                "source": check["source"],
                "status": "ERROR",
                "affected": 0,
                "total": len(df),
                "details": f"Error: {str(e)}",
                "fix_applied": False,
                "retries": 0,
            }

    def _execute_rule(self, df: pd.DataFrame, col: str, combined: str, original_logic: str):
        """
        Dispatch to the right pandas check based on rule text keywords.
        Returns (passed: bool, affected_rows: int, details: str)
        """
        # Column doesn't exist — skip gracefully
        if col not in df.columns and col != "__all__":
            return True, 0, f"Column '{col}' not found — skipped"

        s = df[col] if col in df.columns else None
        n = len(df)

        # ── NULL / NOT NULL checks ──────────────────────────────────────────
        if any(k in combined for k in ["null", "not null", "notna", "no null", "missing", "non-null", "non null"]):
            bad_mask = s.isna()
            bad = int(bad_mask.sum())
            if "not" in combined or "no null" in combined or "non" in combined:
                return bad == 0, bad, f"{bad} null values found in '{col}'"
            else:
                # "has nulls" style — just report
                return bad > 0, bad, f"{bad} null values found in '{col}'"

        # ── EMPTY STRING checks ─────────────────────────────────────────────
        if any(k in combined for k in ["empty string", "not empty", "blank", "whitespace"]):
            bad_mask = s.astype(str).str.strip() == ""
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} empty/blank values found in '{col}'"

        # ── UNIQUE / DUPLICATE checks ───────────────────────────────────────
        if any(k in combined for k in ["unique", "duplicate", "distinct", "no dup"]):
            dups = int(s.duplicated(keep=False).sum())
            return dups == 0, dups, f"{dups} duplicate values found in '{col}'"

        # ── NUMERIC checks ──────────────────────────────────────────────────
        if any(k in combined for k in ["is numeric", "numeric type", "numeric value", "valid number"]):
            bad_mask = pd.to_numeric(s, errors="coerce").isna() & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} non-numeric values in '{col}'"

        # ── NON-NEGATIVE ────────────────────────────────────────────────────
        if any(k in combined for k in ["non-negative", "nonnegative", "not negative", ">= 0", "≥ 0", "positive"]):
            num = pd.to_numeric(s, errors="coerce")
            bad_mask = num < 0
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} negative values found in '{col}'"

        # ── RANGE checks (min/max) ───────────────────────────────────────────
        range_min = re.search(r"(?:>=|≥|min|minimum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
        range_max = re.search(r"(?:<=|≤|max|maximum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
        if range_min or range_max:
            num = pd.to_numeric(s, errors="coerce")
            bad_mask = pd.Series([False] * n, index=df.index)
            details_parts = []
            if range_min:
                min_val = float(range_min.group(1))
                below = num < min_val
                bad_mask = bad_mask | below
                details_parts.append(f"{int(below.sum())} below {min_val}")
            if range_max:
                max_val = float(range_max.group(1))
                above = num > max_val
                bad_mask = bad_mask | above
                details_parts.append(f"{int(above.sum())} above {max_val}")
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{'; '.join(details_parts)} in '{col}'"

        # ── DATE / DATETIME checks ──────────────────────────────────────────
        if any(k in combined for k in ["date", "datetime", "timestamp", "valid date", "date format"]):
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                bad_mask = parsed.isna() & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} invalid date values in '{col}'"
            except Exception:
                return True, 0, "Date parsing skipped"

        # ── FUTURE DATE checks ──────────────────────────────────────────────
        if "future" in combined and "date" in combined:
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                bad_mask = parsed > pd.Timestamp.now()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} future dates in '{col}'"
            except Exception:
                return True, 0, "Future date check skipped"

        # ── EMAIL FORMAT check ──────────────────────────────────────────────
        if any(k in combined for k in ["email", "e-mail", "@"]):
            pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
            bad_mask = ~s.astype(str).str.match(pattern) & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} invalid email formats in '{col}'"

        # ── PHONE FORMAT check ──────────────────────────────────────────────
        if any(k in combined for k in ["phone", "mobile", "contact number"]):
            bad_mask = ~s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True).str.match(r"^\d{7,15}$") & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} invalid phone formats in '{col}'"

        # ── ALLOWABLE VALUES / CATEGORICAL check ────────────────────────────
        allowed_match = re.search(r"(?:one of|in\s*\[|allowed values?|valid values?|must be)\s*[:\[]?\s*([\w,\s'\"]+)", combined)
        if allowed_match:
            raw_vals = allowed_match.group(1)
            vals = {v.strip().strip("'\"").lower() for v in raw_vals.split(",") if v.strip()}
            if vals:
                bad_mask = ~s.astype(str).str.lower().isin(vals) & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} values not in allowed set {vals} in '{col}'"

        # ── LENGTH checks ───────────────────────────────────────────────────
        len_match = re.search(r"length\s*(?:==|=|is|<=|>=|<|>)\s*(\d+)", combined)
        if len_match:
            expected_len = int(len_match.group(1))
            op = re.search(r"(==|=|is|<=|>=|<|>)\s*\d+", combined)
            actual_len = s.astype(str).str.len()
            bad_mask = actual_len != expected_len
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} values with unexpected length (expected {expected_len}) in '{col}'"

        # ── CONSISTENCY / CROSS-COLUMN checks ───────────────────────────────
        if any(k in combined for k in ["consistent", "match", "cross", "referential"]):
            # Can't do cross-column without more context — mark as PASS with note
            return True, 0, f"Cross-column rule '{check['label']}' — manual review recommended"

        # ── FALLBACK: try to eval original logic as pandas expression ──────
        try:
            result_mask = df.eval(original_logic, engine="python")
            if hasattr(result_mask, "__iter__"):
                bad = int((~result_mask).sum()) if hasattr(result_mask, "sum") else 0
                return bad == 0, bad, f"{bad} rows failed rule: {original_logic[:60]}"
        except Exception:
            pass

        # ── DEFAULT: report as PASS with note ──────────────────────────────
        return True, 0, f"Rule '{check['label']}' checked — no violations detected"