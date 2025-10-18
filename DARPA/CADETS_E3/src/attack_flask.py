from flask import Flask, jsonify, request
import json
from pathlib import Path

app = Flask(__name__)

DATA_PATH = Path("../artifact/graph_visual/attack_investigation.json")

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