import math
import os
import re
import itertools
from collections import defaultdict

import pandas as pd
import networkx as nx
import igraph as ig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "Investments Export.csv")
OUT_DIR    = SCRIPT_DIR

# ── FILTER OPTIONS ────────────────────────────────────────────────────────────
# Filter by Company State ("Company State" column). Set to None to include all.
# Examples: "Berlin", "Bayern", "Nordrhein-Westfalen", None
FILTER_STATE = None

# Filter by publication year ("Date Publication" column). Set to None to include all.
# Examples: 2023, 2024, None  —  or a list: [2023, 2024]
FILTER_YEAR  = [2021, 2022, 2023, 2024, 2025]
# ─────────────────────────────────────────────────────────────────────────────
# Build output filename from filter values
_state_tag = "_" + "_".join([FILTER_STATE] if isinstance(FILTER_STATE, str) else FILTER_STATE) if FILTER_STATE else ""
_year_tag  = "_" + "_".join(str(y) for y in ([FILTER_YEAR] if isinstance(FILTER_YEAR, int) else FILTER_YEAR)) if FILTER_YEAR else ""
OUT_PATH   = os.path.join(OUT_DIR, f"top5_investors_composite{_state_tag}{_year_tag}.csv")

# ── 1. Load & clean
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8-sig", dtype=str,
                 engine="python", on_bad_lines="skip")

df["Investor Affiliation"] = df["Investor Affiliation"].fillna("").str.strip()
df = df[df["Investor Affiliation"] != ""]
df = df[~df["Investor Affiliation"].str.contains("rliche Person", na=False)]  # remove "Natürliche Person"

# Apply Company State filter
if FILTER_STATE is not None:
    states = [FILTER_STATE] if isinstance(FILTER_STATE, str) else FILTER_STATE
    df = df[df["Company State"].str.strip().isin(states)]
    print(f"Filter: Company State = {states}")

# Apply publication year filter
if FILTER_YEAR is not None:
    years = [FILTER_YEAR] if isinstance(FILTER_YEAR, int) else FILTER_YEAR
    df["pub_year"] = pd.to_datetime(df["Date Publication"], dayfirst=True, errors="coerce").dt.year
    df = df[df["pub_year"].isin(years)]
    print(f"Filter: Publication Year = {years}")

print(f"Rows after filtering: {len(df):,}")

# ── 2. Parse Shareholder Amount ──────────────────────────────────────────────
def parse_amount(s):
    """Parse German-format amounts like '354744,376 ?' → 354744.376"""
    if not isinstance(s, str) or s.strip() == "":
        return 0.0
    s = re.sub(r"[^\d,]", "", s)   # keep only digits and comma
    s = s.replace(",", ".")         # comma → decimal point
    try:
        return float(s)
    except ValueError:
        return 0.0

df["amount"] = df["Shareholder Amount"].apply(parse_amount)
df["shares"] = df["Shareholder New Shares"].apply(parse_amount)

# ── 3. Deal key (one round = same company + same publication date) ────────────
df["round_key"] = df["Company_ID"].astype(str) + "||" + df["Date Publication"].astype(str)

# ── 4. Build amount-weighted co-investment graph ─────────────────────────────
print("Building co-investment graph...")
edge_weight = defaultdict(float)  # total capital deployed together

for _, grp in df.groupby("round_key"):
    inv_amounts = grp.groupby("Investor Affiliation")["amount"].sum().to_dict()
    investors = sorted(inv_amounts.keys())
    if len(investors) < 2:
        continue
    for a, b in itertools.combinations(investors, 2):
        edge_weight[(a, b)] += inv_amounts[a] + inv_amounts[b]

G = nx.Graph()
G.add_nodes_from(sorted(df["Investor Affiliation"].unique()))
for (a, b), w in edge_weight.items():
    G.add_edge(a, b, weight=w)

print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")

# ── 5. Network metrics (igraph C backend for speed) ──────────────────────────
print("Converting graph to igraph...")
g = ig.Graph.from_networkx(G)
_names = g.vs["_nx_name"]  # node names indexed by vertex id

print("Calculating amount-weighted PageRank...")
_pr = g.pagerank(weights="weight", damping=0.85)
pagerank = dict(zip(_names, _pr))

print("Calculating betweenness centrality (unweighted — bridge detection)...")
_bt = g.betweenness(directed=False)  # exact, no sampling needed with C backend
_n = g.vcount()
_norm = (_n - 1) * (_n - 2) / 2 if _n > 2 else 1.0  # NetworkX-compatible normalisation
betweenness = dict(zip(_names, [b / _norm for b in _bt]))

print("Calculating Burt's Constraint (structural holes)...")
_bc = g.constraint(weights="weight")
burt_constraint = {name: (v if v == v else 1.0) for name, v in zip(_names, _bc)}

print("Detecting communities (Louvain) for Participation Coefficient...")
_communities_ig = g.community_multilevel(weights="weight")
comm_map = {_names[v]: c for v, c in enumerate(_communities_ig.membership)}

def _participation_coefficient(G, comm_map, weight="weight"):
    result = {}
    for node in G.nodes():
        k_i = sum(d.get(weight, 1) for _, d in G[node].items())
        if k_i == 0:
            result[node] = 0.0
            continue
        comm_w = defaultdict(float)
        for neighbor, data in G[node].items():
            comm_w[comm_map[neighbor]] += data.get(weight, 1)
        result[node] = 1.0 - sum((w / k_i) ** 2 for w in comm_w.values())
    return result

participation = _participation_coefficient(G, comm_map)
print(f"  Communities detected: {len(_communities_ig)}")

# ── 6. Fundamental investor stats
print("Calculating fundamental stats...")
stats = (
    df.groupby("Investor Affiliation")
    .agg(
        total_capital    = ("amount",     "sum"),
        unique_companies = ("Company_ID", "nunique"),
        num_investments  = ("Investment_ID", "nunique"),
    )
    .reset_index()
)

# Average round size (total capital raised per round, averaged across investor's rounds)
round_totals = df.groupby("round_key")["amount"].sum().rename("round_total")
df2 = df.join(round_totals, on="round_key")
avg_round = (
    df2.groupby("Investor Affiliation")["round_total"]
    .mean()
    .rename("avg_round_size")
    .reset_index()
)
stats = stats.merge(avg_round, on="Investor Affiliation")

# Lead investor: highest Shareholder New Shares per round
# Sum shares per investor per round (in case of multiple rows)
shares_per_round = (
    df.groupby(["round_key", "Investor Affiliation"])["shares"]
    .sum()
    .reset_index()
)
# For each round, find the max shares value
round_max_shares = shares_per_round.groupby("round_key")["shares"].transform("max")
# Mark rows where investor has max shares in that round (and shares > 0)
shares_per_round["is_lead"] = (
    (shares_per_round["shares"] == round_max_shares) &
    (shares_per_round["shares"] > 0)
)
lead_counts = (
    shares_per_round[shares_per_round["is_lead"]]
    .groupby("Investor Affiliation")["round_key"]
    .nunique()
    .rename("lead_rounds")
    .reset_index()
)
stats = stats.merge(lead_counts, on="Investor Affiliation", how="left")
stats["lead_rounds"] = stats["lead_rounds"].fillna(0)

# ── 7. Combine & normalise ───────────────────────────────────────────────────
stats["pagerank"]       = stats["Investor Affiliation"].map(pagerank)
stats["betweenness"]    = stats["Investor Affiliation"].map(betweenness)
# Burt: invert so that HIGH score = many structural holes (good)
stats["burt_holes"]     = 1.0 - stats["Investor Affiliation"].map(burt_constraint)
stats["participation"]  = stats["Investor Affiliation"].map(participation)

def minmax(series):
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo) if hi > lo else series * 0.0

metrics = ["pagerank", "betweenness", "burt_holes",
           "participation", "total_capital", "unique_companies", "avg_round_size", "lead_rounds"]
for m in metrics:
    stats[f"{m}_norm"] = minmax(stats[m])

# ── Composite weights (total = 1.0) ───────────────────────────────────────────
#
#  Netzwerk-Zentralität       (47%)
#    PageRank          0.20  — Position im Netzwerk, kapitalgewichtet
#    Betweenness       0.15  — Brückenfunktion zwischen Clustern
#    Burt's Holes      0.12  — Strukturelle Lücken / Informationsvorteil
#
#  Community-Vernetzung       (18%)
#    Participation     0.18  — Verbindungen über Community-Grenzen hinweg
#
#  Fundamentaldaten           (35%)
#    Total Capital     0.12  — Gesamtkapital
#    Unique Companies  0.10  — Portfoliobreite
#    Avg Round Size    0.06  — Deal-Qualität
#    Lead Rounds       0.07  — Wie oft war Investor Lead (höchste Anteilsanzahl)
#
WEIGHTS = {
    "pagerank_norm":        0.20,
    "betweenness_norm":     0.15,
    "burt_holes_norm":      0.12,
    "participation_norm":   0.18,
    "total_capital_norm":   0.12,
    "unique_companies_norm":0.10,
    "avg_round_size_norm":  0.06,
    "lead_rounds_norm":     0.07,
}
stats["composite_score"] = sum(stats[col] * w for col, w in WEIGHTS.items())

# ── 8. Top 5% ────────────────────────────────────────────────────────────────
stats = stats.sort_values("composite_score", ascending=False).reset_index(drop=True)
n     = len(stats)
top_k = max(1, math.ceil(n * 0.05))
top   = stats.head(top_k).copy()

# Pretty-print columns
display_cols = [
    "Investor Affiliation",
    "composite_score",
    "pagerank",
    "betweenness",
    "burt_holes",
    "participation",
    "total_capital",
    "unique_companies",
    "avg_round_size",
    "lead_rounds",
    "num_investments",
]

print(f"\nTotal investor affiliations : {n}")
print(f"Top 5%                      : {top_k}\n")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.float_format", "{:.6f}".format)
print(top[display_cols].to_string(index=True))

# ── 9. Save ──────────────────────────────────────────────────────────────────
top[display_cols].to_csv(OUT_PATH, index=False, sep=";")
print(f"\nSaved → {OUT_PATH}")
