from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from typing import cast
from pathlib import Path
from typing import List
import os

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Data source: local Excel file with price per sqm (USD)
EXCEL_CANDIDATES: List[str] = [
    "price_per_sqm_full.xlsx",
    "price_per_sqm_full.xls",
    str(Path("data") / "price_per_sqm_full.xlsx"),
    str(Path("data") / "price_per_sqm_full.xls"),
]

# Simple in-memory cache for the DataFrame
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours
_df_cache: Optional[pd.DataFrame] = None
_df_cached_at: float = 0.0


@dataclass(frozen=True)
class LookupResult:
    city: str
    country: str
    price_per_sqm_usd: float


# --- Validation helpers ---

CITY_COUNTRY_RE = re.compile(r"^[A-Za-zÀ-ÿ'’.\- ]+$")  # allow accents, spaces, hyphen, apostrophes, dot


def validate_sqm(raw: str) -> Tuple[Optional[int], Optional[str]]:
    raw = (raw or "").strip()
    if not raw:
        return None, "Square meters is required."
    if not raw.isdigit():
        return None, "Square meters must be an integer."
    sqm = int(raw)
    if sqm <= 0:
        return None, "Square meters must be greater than 0."
    return sqm, None


def validate_text_field(raw: str, field_name: str) -> Tuple[Optional[str], Optional[str]]:
    val = (raw or "").strip()
    if not val:
        return None, f"{field_name} is required."
    if not CITY_COUNTRY_RE.match(val):
        return None, f"{field_name} must be a text string (letters/spaces only)."
    return val, None


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _find_excel_path() -> Optional[Path]:
    for candidate in EXCEL_CANDIDATES:
        p = Path(candidate)
        if p.exists() and p.is_file():
            return p
    return None


def _load_price_df() -> pd.DataFrame:
    global _df_cache, _df_cached_at
    now = time.time()
    if _df_cache is not None and (now - _df_cached_at) < _CACHE_TTL_SECONDS:
        return _df_cache

    excel_path = _find_excel_path()
    if not excel_path:
        raise FileNotFoundError(
            "Could not find 'price_per_sqm_full' Excel file. Expected one of: "
            + ", ".join(EXCEL_CANDIDATES)
        )

    df = pd.read_excel(excel_path)
    # Normalize column names for matching
    df.columns = [normalize(str(c)) for c in df.columns]

    # Identify likely columns
    def pick(col_opts: List[str]) -> Optional[str]:
        for c in df.columns:
            for opt in col_opts:
                if opt in c:
                    return c
        return None

    city_col = pick(["city"]) or "city"
    country_col = pick(["country"]) or "country"
    price_col = pick(["price per", "price_per", "usd", "sqm", "m2", "square"]) or "price_per_sqm_usd"

    # Validate presence
    for required in [city_col, country_col, price_col]:
        if required not in df.columns:
            raise ValueError(
                "Excel is missing a required column. Need city/country/price columns. "
                "Found columns: " + ", ".join(df.columns)
            )

    # Keep only required columns under canonical names
    df = df[[city_col, country_col, price_col]].rename(
        columns={city_col: "city", country_col: "country", price_col: "price"}
    )

    # Drop rows without price
    df = df.dropna(subset=["price"]).copy()

    # Normalize lookup keys
    df["_city_key"] = df["city"].astype(str).map(normalize)
    df["_country_key"] = df["country"].astype(str).map(normalize)

    _df_cache = df
    _df_cached_at = now
    return df


def lookup_price_for_city_country(city: str, country: str) -> Optional[LookupResult]:
    df = _load_price_df()
    c_key = normalize(city)
    k_key = normalize(country)

    # Exact match on normalized keys
    matches = df[(df["_city_key"] == c_key) & (df["_country_key"] == k_key)]
    if matches.empty:
        return None
    # If multiple rows, pick the first
    row = matches.iloc[0]
    try:
        price = float(row["price"])
    except Exception:
        return None
    return LookupResult(city=str(row["city"]), country=str(row["country"]), price_per_sqm_usd=price)


# --- Routes ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/estimate", methods=["POST"])
def estimate():
    sqm, sqm_err = validate_sqm(request.form.get("sqm", ""))
    city, city_err = validate_text_field(request.form.get("city", ""), "City")
    country, country_err = validate_text_field(request.form.get("country", ""), "Country")

    errors = [e for e in [sqm_err, city_err, country_err] if e]
    if errors:
        return render_template("index.html", errors=errors, form=request.form), 400

    try:
        city_s = cast(str, city)
        country_s = cast(str, country)
        sqm_i = cast(int, sqm)
        result = lookup_price_for_city_country(city_s, country_s)
    except FileNotFoundError as e:
        return render_template(
            "index.html",
            errors=[str(e)],
            form=request.form,
        ), 500
    except ValueError as e:
        return render_template(
            "index.html",
            errors=[str(e)],
            form=request.form,
        ), 500

    if not result:
        return render_template(
            "index.html",
            errors=[f"City/country not found in dataset: '{city}, {country}'."],
            form=request.form,
        ), 404

    assert result is not None
    total = sqm_i * result.price_per_sqm_usd
    return render_template(
        "result.html",
        sqm=sqm,
        city=result.city,
        country=result.country,
        price_per_sqm=result.price_per_sqm_usd,
        total_price=total,
    )


if __name__ == "__main__":
    # Run: python app.py
    app.run(debug=True)