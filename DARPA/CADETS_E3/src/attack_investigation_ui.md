# Enriched Attack Investigation UI Blueprint

## Overview
This document captures a reference UX for exploring enriched attack graphs produced by `attack_investigation.py`. The UI consumes the JSON emitted at `artifact_dir/graph_visual/attack_investigation.json` and exposes search, filtering, timeline, and visualization workflows. The UI artifact is a Markdown design spec (attack_investigation_ui.md). If you implement the reference code inside it, you’d save the backend portion as a Python Flask module (depends on Flask) and the front-end snippet as a React/TypeScript file (depends on React plus react-force-graph-2d, which wraps D3/Three).

## API Endpoints (Flask Reference)

```python
from flask import Flask, jsonify, request
import json
from pathlib import Path

app = Flask(__name__)
DATA_PATH = Path("<ARTIFACT_DIR>/graph_visual/attack_investigation.json")

def load_graph_payload():
    return json.loads(DATA_PATH.read_text())

@app.get("/api/alerts")
def list_alerts():
    payload = load_graph_payload()
    min_risk = int(request.args.get("min_risk", 0))
    communities = payload["communities"]
    nodes = {node["id"]: node for node in payload["nodes"]}

    alerts = []
    for node_id, community_id in communities.items():
        node = nodes.get(node_id, {})
        if node.get("risk_score", 0) < min_risk:
            continue
        alerts.append(
            {
                "id": f"{community_id}-{node_id}",
                "community": community_id,
                "node": node,
            }
        )
    return jsonify({"alerts": alerts})

@app.get("/api/graph")
def get_graph():
    return jsonify(load_graph_payload())

@app.get("/api/search")
def search_entities():
    query = request.args.get("q", "").lower()
    payload = load_graph_payload()
    matches = [
        node for node in payload["nodes"]
        if query in (node.get("label") or "").lower()
        or query in (node.get("cmdline") or "").lower()
    ]
    return jsonify({"results": matches})
```

## Front-End Structure (D3 + React Suggested)

```tsx
import React, { useEffect, useState } from "react";
import ForceGraph2D, { ForceGraphMethods } from "react-force-graph-2d";

export const AttackGraphViewer: React.FC = () => {
  const [graph, setGraph] = useState<any>({ nodes: [], links: [] });
  const fgRef = React.useRef<ForceGraphMethods>();

  useEffect(() => {
    fetch("/api/graph")
      .then((res) => res.json())
      .then((payload) =>
        setGraph({
          nodes: payload.nodes.map((node: any) => ({
            id: node.id,
            label: node.label,
            risk: node.risk_score,
            attack: node.attack_flag,
            metadata: node,
          })),
          links: payload.edges.map((edge: any) => ({
            source: edge.source,
            target: edge.target,
            importance: edge.importance_score,
            risk: edge.risk_score,
            metadata: edge,
          })),
        })
      );
  }, []);

  return (
    <div className="graph-shell">
      <ForceGraph2D
        ref={fgRef}
        graphData={graph}
        nodeLabel={(node: any) =>
          `${node.label}\nRisk: ${node.risk}\nAttack: ${node.attack}`
        }
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const label = node.label || node.id;
          const fontSize = 12 / globalScale;
          const color =
            node.attack ? "#ff4444" : node.risk >= 70 ? "#ff9900" : "#1f77b4";
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x!, node.y!, 6, 0, 2 * Math.PI, false);
          ctx.fill();
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.fillStyle = "rgba(0,0,0,0.7)";
          ctx.fillText(label, node.x! + 8, node.y! + 2);
        }}
        linkColor={(link: any) =>
          link.importance >= 70 ? "#d62728" : "#1f77b4"
        }
      />
    </div>
  );
};
```

## UX Highlights

- **Alert Sidebar**: Lists communities sorted by highest risk, surface attack-flagged entities, and provide quick filters (`min_risk`, `entity_type`, `time range`).
- **Graph Canvas**: Force-directed visualization with risk-based coloring, hover tooltips, and click-to-expand metadata drawer.
- **Timeline Panel**: Shows temporal activity using `temporal_features.mean_interval`, `burst_score`, allowing pivot between normal vs. bursty periods.
- **Search & IOC Matching**: `/api/search` supports fuzzy queries, while UI highlights matches directly on graph.
- **Export & Collaboration**: Download subgraph JSON/PDF, add analyst annotations, and share investigation bundles.

## Implementation Checklist

1. Serve JSON via lightweight Flask API (or static hosting with pre-generated payload).
2. Front-end fetches enriched graph, renders interactive visualization, and binds filters.
3. Persist analyst interactions (tags, notes) in dedicated store (SQLite or JSON).
4. Integrate authentication/authorization as needed for SOC deployment.

Replace `<ARTIFACT_DIR>` with your absolute artifact directory path.