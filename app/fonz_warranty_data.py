"""
Shared helpers for Fonz warranty Excel → JSON → lookup / FAISS ingest.

Source workbook: ``raw_data/Fonz's All-in-one Warranty List.xlsx``
Outputs:
  - ``data/fonz_error_codes.json``
  - ``data/fonz_model_diagnostics.json``
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONZ_ERROR_CODES_PATH = _PROJECT_ROOT / "data" / "fonz_error_codes.json"
FONZ_MODEL_DIAG_PATH = _PROJECT_ROOT / "data" / "fonz_model_diagnostics.json"
DEFAULT_XLSX_PATH = (
    _PROJECT_ROOT / "raw_data" / "Fonz's All-in-one Warranty List.xlsx"
)

_REF_RE = re.compile(r"#REF!", re.I)


def normalize_model_key(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"\bos-", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def normalize_error_code(text: str) -> str:
    s = (text or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def expand_error_codes(raw: str) -> list[str]:
    """
    Expand Fonz-style code cells into discrete lookup keys.

    Examples:
      ``C6`` → [``C6``]
      ``C1 - C5`` → [``C1``, ``C2``, ``C3``, ``C4``, ``C5``]
      ``CA / CB`` → [``CA``, ``CB``]
      ``E1 - E3`` → [``E1``, ``E2``, ``E3``]
    """
    s = (raw or "").strip()
    if not s or s.lower() == "nan":
        return []

    codes: list[str] = []
    for segment in re.split(r"\s*/\s*", s):
        segment = segment.strip()
        if not segment:
            continue
        compact = re.sub(r"\s+", "", segment)
        range_m = re.match(r"^([A-Za-z]*)(\d+)\s*-\s*([A-Za-z]*)(\d+)$", compact)
        if range_m:
            prefix1, start_s, prefix2, end_s = range_m.groups()
            prefix = (prefix1 or prefix2 or "").upper()
            start, end = int(start_s), int(end_s)
            if end >= start and end - start <= 25:
                for num in range(start, end + 1):
                    codes.append(f"{prefix}{num}")
            continue
        codes.append(normalize_error_code(segment))

    # Preserve order, drop dupes.
    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def _clean_cell(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if _REF_RE.search(text):
        return ""
    return text


def build_faiss_page_content(entry: dict[str, Any]) -> str:
    return (
        f"[Source]: Fonz Warranty List\n"
        f"[Model]: {entry.get('model', '')}\n"
        f"[Manufacturer]: {entry.get('manufacturer', '')}\n"
        f"[Category]: {entry.get('workflow_category', '')}\n"
        f"[Error Code]: {entry.get('error_code', '')}\n"
        f"[Meaning]: {entry.get('meaning', '')}\n"
        f"[Troubleshooting]: {entry.get('troubleshooting', '')}"
    ).strip()


def infer_workflow_category(meaning: str, troubleshooting: str = "") -> str:
    """Map Fonz row text to workflow/knowledge category (power, air, mech, …)."""
    try:
        from warranty_knowledge import _infer_category  # noqa: WPS433

        return _infer_category(f"{meaning} {troubleshooting}".strip())
    except Exception:
        return "general"


def load_error_code_records(*, path: Path | None = None) -> list[dict[str, Any]]:
    codes_path = path or FONZ_ERROR_CODES_PATH
    if not codes_path.is_file():
        return []
    try:
        with codes_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def load_model_diagnostic_records() -> list[dict[str, Any]]:
    if not FONZ_MODEL_DIAG_PATH.is_file():
        return []
    try:
        with FONZ_MODEL_DIAG_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    models = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict)]


def fonz_faiss_documents(*, error_codes_path: Path | None = None) -> list[Any]:
    """LangChain Document list for freshdesk_qa index (optional import)."""
    try:
        from langchain_core.documents import Document
    except ImportError:
        return []

    docs: list[Any] = []
    for entry in load_error_code_records(path=error_codes_path):
        content = build_faiss_page_content(entry)
        if len(content) < 40:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "type": "error_code",
                    "source": "fonz",
                    "model": entry.get("model", ""),
                    "model_key": entry.get("model_key", ""),
                    "error_code": entry.get("error_code", ""),
                    "manufacturer": entry.get("manufacturer", ""),
                },
            )
        )
    return docs


def ingest_workbook(
    xlsx_path: Path,
    *,
    error_out: Path = FONZ_ERROR_CODES_PATH,
    diag_out: Path = FONZ_MODEL_DIAG_PATH,
) -> dict[str, int]:
    import pandas as pd

    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    err_df = pd.read_excel(xlsx_path, sheet_name="Error Code Log")
    summary_df = pd.read_excel(xlsx_path, sheet_name="Summary by Mfg & Model")
    map_df = pd.read_excel(xlsx_path, sheet_name="Map", header=3)

    map_df.columns = [str(c).strip() for c in map_df.columns]
    drive_by_model: dict[str, str] = {}
    for _, row in map_df.iterrows():
        model = _clean_cell(row.get("OTA Model Name"))
        drive = _clean_cell(row.get("Drive Folder Link"))
        if model and drive:
            drive_by_model[normalize_model_key(model)] = drive

    error_entries: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for _, row in err_df.iterrows():
        model = _clean_cell(row.get("Chair Model"))
        manufacturer = _clean_cell(row.get("Manufacturer"))
        raw_code = _clean_cell(row.get("Error Code"))
        meaning = _clean_cell(row.get("Meaning"))
        troubleshooting = _clean_cell(row.get("Troubleshooting Steps"))
        parts = _clean_cell(row.get("Parts Required"))
        severity = _clean_cell(row.get("Severity"))

        if not model or not raw_code:
            continue
        if not meaning and not troubleshooting:
            continue

        model_key = normalize_model_key(model)
        for code in expand_error_codes(raw_code):
            key = (model_key, code)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            error_entries.append(
                {
                    "manufacturer": manufacturer,
                    "model": model,
                    "model_key": model_key,
                    "error_code": code,
                    "error_code_raw": raw_code,
                    "meaning": meaning,
                    "troubleshooting": troubleshooting,
                    "parts_required": parts,
                    "severity": severity,
                    "workflow_category": infer_workflow_category(meaning, troubleshooting),
                }
            )

    diag_models: list[dict[str, Any]] = []
    seen_models: set[str] = set()

    for _, row in summary_df.iterrows():
        model = _clean_cell(row.get("OTA Model Name"))
        if not model:
            continue
        model_key = normalize_model_key(model)
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        diag_models.append(
            {
                "mfg": _clean_cell(row.get("Mfg")),
                "mfg_product_code": _clean_cell(row.get("Mfg Product Code")),
                "model": model,
                "model_key": model_key,
                "total_error_codes": _clean_cell(row.get("Total Error Codes")),
                "entry_method": _clean_cell(row.get("Entry Method?")),
                "entry_procedure": _clean_cell(row.get("Entry Procedure")),
                "error_code_access_type": _clean_cell(row.get("Error Code Access Type")),
                "drive_folder_url": drive_by_model.get(model_key, ""),
            }
        )

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_out.parent.mkdir(parents=True, exist_ok=True)
    diag_out.parent.mkdir(parents=True, exist_ok=True)

    with error_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "1.0",
                "updated": updated,
                "source_file": xlsx_path.name,
                "entry_count": len(error_entries),
                "entries": error_entries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with diag_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "1.0",
                "updated": updated,
                "source_file": xlsx_path.name,
                "model_count": len(diag_models),
                "models": diag_models,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "error_code_entries": len(error_entries),
        "model_diagnostics": len(diag_models),
    }
