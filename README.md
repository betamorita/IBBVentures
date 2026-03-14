# IBBVentures

Python scripts for analyzing startup financing rounds and investor networks in Germany.

## Scripts

### `investor_composite_score.py`
Ranks investors by a weighted composite score combining network centrality metrics and fundamental investment data.

**Network metrics (50%):** PageRank (capital-weighted), betweenness centrality (bridge detection), Burt's structural holes, clustering coefficient (network diversity).

**Community connectivity (15%):** Participation coefficient — how much an investor connects across different community clusters.

**Fundamentals (35%):** Total capital deployed, portfolio breadth, average round size, lead investor frequency.

Output: Top 5% of investors by composite score.

### `visualize_network.py`
Generates an interactive HTML visualization of the co-investment network (top 5% by betweenness centrality) using pyvis. Node size reflects betweenness, node color reflects PageRank, and edge width reflects co-investment frequency.

### `create_filtered_financing_rounds.py`
Exports a filtered list of financing rounds with aggregated investor lists per round. Supports filtering by:
- **Minimum financing amount** (total shareholder amount per round)
- **Company state** (e.g. Berlin, Bayern)
- **Publication year(s)**

Output includes round-level details (company, date, capital, industry, etc.), total financing, participating investors, and amount source.

## Data Source
The raw investment data used by these scripts can be downloaded at [addedval.io](https://addedval.io) (subscription required). Place the exported CSV (`Investments Export.csv`) in the same directory as the scripts.

## Requirements
```
pandas
networkx
pyvis  # only for visualize_network.py
```

## Usage
Configure the filter options at the top of each script, then run:
```
python <script_name>.py
```
All input/output paths are relative to the script directory.
