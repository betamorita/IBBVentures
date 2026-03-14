import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "Investments Export.csv")
OUT_DIR  = SCRIPT_DIR

# ── FILTER OPTIONS ────────────────────────────────────────────────────────────
# Minimum total financing per round (sum of Shareholder Amount in EUR).
# Set to 0 or None to include all rounds.
FILTER_MIN_FINANCING = 5_000_000

# Filter by Company State. Set to None to include all.
# Examples: "Berlin", ["Berlin", "Bayern"], None
FILTER_STATE = "Berlin"

# Filter by publication year. Set to None to include all.
# Examples: 2025, [2024, 2025], None
FILTER_YEAR = [2021, 2022, 2023, 2024, 2025, 2026]
# ──────────────────────────────────────────────────────────────────────────────

# ── Build output filename from filter values ─────────────────────────────────
_fin_tag   = f"_min{FILTER_MIN_FINANCING}" if FILTER_MIN_FINANCING else ""
_state_tag = "_" + "_".join([FILTER_STATE] if isinstance(FILTER_STATE, str) else FILTER_STATE) if FILTER_STATE else ""
_year_tag  = "_" + "_".join(str(y) for y in ([FILTER_YEAR] if isinstance(FILTER_YEAR, int) else FILTER_YEAR)) if FILTER_YEAR else ""
OUT_PATH   = os.path.join(OUT_DIR, f"filtered_financing_rounds{_fin_tag}{_state_tag}{_year_tag}.csv")

# ── 1. Load & clean ──────────────────────────────────────────────────────────
print("Loading CSV ...")
df = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8-sig", dtype=str,
                 engine="python", on_bad_lines="skip")

# ── 2. Apply Company State filter ────────────────────────────────────────────
if FILTER_STATE is not None:
    states = [FILTER_STATE] if isinstance(FILTER_STATE, str) else FILTER_STATE
    df = df[df["Company State"].str.strip().isin(states)]
    print(f"Filter: Company State = {states}")

# ── 3. Apply publication year filter ─────────────────────────────────────────
if FILTER_YEAR is not None:
    years = [FILTER_YEAR] if isinstance(FILTER_YEAR, int) else FILTER_YEAR
    df["pub_year"] = pd.to_datetime(df["Date Publication"], dayfirst=True, errors="coerce").dt.year
    df = df[df["pub_year"].isin(years)]
    print(f"Filter: Publication Year = {years}")

print(f"Rows after pre-filtering: {len(df):,}")

# ── 4. Parse currency amounts ────────────────────────────────────────────────
def parse_amount(s):
    """Parse German-format amounts like '1.116.205 €' → 1116205.0"""
    if not isinstance(s, str) or s.strip() == "":
        return 0.0
    s = re.sub(r"[^\d,.]", "", s)   # keep digits, comma, dot
    # German format: dots are always thousand separators, comma is decimal
    s = s.replace(".", "")          # remove thousand separators
    s = s.replace(",", ".")         # comma → decimal point
    try:
        return float(s)
    except ValueError:
        return 0.0

df["shareholder_amount_num"] = df["Shareholder Amount"].apply(parse_amount)

# ── 5. Define round key & aggregate per round ────────────────────────────────
df["round_key"] = df["Company_ID"].astype(str) + "||" + df["Date Publication"].astype(str)

# Total financing per round (sum of all shareholder amounts)
round_financing = df.groupby("round_key")["shareholder_amount_num"].sum().rename("total_financing")

# ── 6. Apply minimum financing filter ────────────────────────────────────────
if FILTER_MIN_FINANCING:
    qualifying_rounds = round_financing[round_financing >= FILTER_MIN_FINANCING].index
    df = df[df["round_key"].isin(qualifying_rounds)]
    print(f"Filter: Minimum Financing = {FILTER_MIN_FINANCING:,.0f} EUR")

print(f"Rows after all filters: {len(df):,}")

if df.empty:
    print("No rounds match the given filters.")
    exit()

# ── 7. Aggregate investors and amount sources per round ──────────────────────
investors_agg = (
    df.groupby("round_key")["Investor Affiliation"]
    .apply(lambda x: " | ".join(sorted(x.dropna().str.strip().unique())))
    .rename("Investors")
)

amount_source_agg = (
    df.groupby("round_key")["Amount Source"]
    .apply(lambda x: x.dropna().str.strip().mode().iloc[0] if not x.dropna().empty else "")
    .rename("Amount Source")
)

# ── 8. Select round-level columns (one row per round) ────────────────────────
round_cols = [
    "round_key",
    "Company_ID",
    "Current Company Name",
    "Company State",
    "Date Publication",
    "Round Nr",
    "Increased By",
    "New Capital",
    "Date Founded",
    "Current Company Status",
    "Industry",
    "Customer Focus",
    "Business Model",
]

rounds = df[round_cols].drop_duplicates(subset=["round_key"]).copy()

# Join total financing, investor list & amount sources
rounds = rounds.merge(round_financing, left_on="round_key", right_index=True)
rounds = rounds.merge(investors_agg, left_on="round_key", right_index=True)
rounds = rounds.merge(amount_source_agg, left_on="round_key", right_index=True)

# Format total financing for readability
rounds["total_financing"] = rounds["total_financing"].apply(lambda x: f"{x:,.2f}")

# Sort by total financing descending (parse back for sorting)
rounds["_sort"] = rounds["total_financing"].str.replace(",", "").astype(float)
rounds = rounds.sort_values("_sort", ascending=False).drop(columns=["_sort", "round_key"])
rounds = rounds.reset_index(drop=True)

# ── 9. Output ─────────────────────────────────────────────────────────────────
print(f"\nFinancing rounds found: {len(rounds):,}\n")

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 220)
print(rounds.head(20).to_string(index=False))

rounds.to_csv(OUT_PATH, index=False, sep=";", encoding="utf-8-sig")
print(f"\nSaved → {OUT_PATH}")
