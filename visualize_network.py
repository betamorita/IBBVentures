import re
import math
import os
import itertools
from collections import defaultdict

import pandas as pd
import networkx as nx
from pyvis.network import Network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "Investors_Berlin.csv")
OUT_HTML = os.path.join(SCRIPT_DIR, "investor_network.html")

# ── 1. Load & clean ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1", dtype=str)
df["Investor Affiliation"] = df["Investor Affiliation"].fillna("").str.strip()
df = df[df["Investor Affiliation"] != ""]
df = df[~df["Investor Affiliation"].str.contains("rliche Person", na=False)]

def parse_amount(s):
    if not isinstance(s, str) or s.strip() == "":
        return 0.0
    s = re.sub(r"[^\d,]", "", s)
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0

df["amount"] = df["Shareholder Amount"].apply(parse_amount)
df["round_key"] = df["Company_ID"].astype(str) + "||" + df["Date Publication"].astype(str)

# ── 2. Build graph ───────────────────────────────────────────────────────────
print("Building graph...")
edge_weight = defaultdict(float)
edge_count  = defaultdict(int)

for _, grp in df.groupby("round_key"):
    inv_amounts = grp.groupby("Investor Affiliation")["amount"].sum().to_dict()
    investors = sorted(inv_amounts.keys())
    if len(investors) < 2:
        continue
    for a, b in itertools.combinations(investors, 2):
        edge_weight[(a, b)] += inv_amounts[a] + inv_amounts[b]
        edge_count[(a, b)]  += 1

G = nx.Graph()
G.add_nodes_from(sorted(df["Investor Affiliation"].unique()))
for (a, b), w in edge_weight.items():
    G.add_edge(a, b, weight=w, count=edge_count[(a, b)])

# ── 3. Compute metrics ───────────────────────────────────────────────────────
print("Computing metrics...")
betweenness = nx.betweenness_centrality(G, normalized=True)
pagerank    = nx.pagerank(G, weight="weight", max_iter=1000, alpha=0.85)

total_capital = df.groupby("Investor Affiliation")["amount"].sum().to_dict()

# ── 4. Filter to top 5% by betweenness ──────────────────────────────────────
n      = len(betweenness)
top_k  = max(1, math.ceil(n * 0.05))
top_investors = set(
    sorted(betweenness, key=betweenness.get, reverse=True)[:top_k]
)
print(f"Visualising top {top_k} of {n} investors...")

sub = G.subgraph(top_investors).copy()
# Also keep edges that cross into the subgraph for context? No — keep it clean.

# ── 5. Normalise for visual scaling ─────────────────────────────────────────
def minmax_scale(values, out_min, out_max):
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: (out_min + out_max) / 2 for k in values}
    return {k: out_min + (v - lo) / (hi - lo) * (out_max - out_min)
            for k, v in values.items()}

bw_sub  = {n: betweenness[n] for n in sub.nodes}
pr_sub  = {n: pagerank[n]    for n in sub.nodes}
cap_sub = {n: total_capital.get(n, 0) for n in sub.nodes}

node_size  = minmax_scale(bw_sub,  15, 60)   # size  = betweenness
node_color_val = minmax_scale(pr_sub, 0, 1)  # color = pagerank (blue→red)

def pr_to_color(v):
    # low pagerank → blue (#4e79a7), high → red (#e15759)
    r = int(78  + (225 - 78)  * v)
    g = int(121 + (87  - 121) * v)
    b = int(167 + (89  - 167) * v)
    return f"#{r:02x}{g:02x}{b:02x}"

# ── 6. Build pyvis network ───────────────────────────────────────────────────
net = Network(
    height="820px",
    width="100%",
    bgcolor="#1a1a2e",
    font_color="white",
    notebook=False,
)
net.force_atlas_2based(
    gravity=-60,
    central_gravity=0.01,
    spring_length=120,
    spring_strength=0.05,
    damping=0.4,
)

for node in sub.nodes:
    cap_m = cap_sub[node] / 1_000_000
    bw_pct = bw_sub[node] * 100
    pr_val = pr_sub[node]
    tooltip = (
        f"<b>{node}</b><br>"
        f"Betweenness: {bw_pct:.2f}%<br>"
        f"PageRank:    {pr_val:.5f}<br>"
        f"Total Capital: €{cap_m:.1f}M"
    )
    net.add_node(
        node,
        label=node,
        size=node_size[node],
        color=pr_to_color(node_color_val[node]),
        title=tooltip,
        font={"size": 10, "color": "white"},
    )

# Edge width = log-scaled co-investment count
all_counts = [sub[u][v]["count"] for u, v in sub.edges]
if all_counts:
    max_c = max(all_counts)
    for u, v, data in sub.edges(data=True):
        width = 1 + (data["count"] / max_c) * 6
        w_m   = data["weight"] / 1_000_000
        net.add_edge(
            u, v,
            width=width,
            color="rgba(180,180,180,0.25)",
            title=f"Co-investments: {data['count']}<br>Combined capital: €{w_m:.1f}M",
        )

# ── 7. Legend as HTML ────────────────────────────────────────────────────────
legend_html = """
<div style="position:fixed;top:15px;left:15px;background:#222;padding:12px 16px;
            border-radius:8px;color:white;font-family:sans-serif;font-size:13px;z-index:999;">
  <b>Berlin Investor Network — Top 5%</b><br><br>
  <span style="color:#aaa">Node size</span>  → Betweenness centrality<br>
  <span style="color:#aaa">Node color</span> → PageRank
    (<span style="color:#4e79a7">■</span> low &nbsp;→&nbsp; <span style="color:#e15759">■</span> high)<br>
  <span style="color:#aaa">Edge width</span> → Co-investment frequency<br><br>
  <span style="color:#aaa;font-size:11px">Hover nodes/edges for details</span>
</div>
"""

net.save_graph(OUT_HTML)

# Inject legend into saved HTML
with open(OUT_HTML, "r", encoding="utf-8") as f:
    html = f.read()
html = html.replace("<body>", f"<body>\n{legend_html}", 1)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nDone! Open in browser:\n{OUT_HTML}")
