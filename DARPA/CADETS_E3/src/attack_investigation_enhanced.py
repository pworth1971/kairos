#!/usr/bin/env python3
"""
Attack Graph Enrichment, Community Detection, and Visualization.

This script ingests anomalous edge logs produced by anomalous_queue_construction.py,
builds a directed provenance graph, enriches nodes/edges with contextual metadata,
runs community detection, exports Graphviz PDFs, and emits a JSON artifact that can
feed richer investigation UIs.
"""
import ast
import json
import os
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from community import community_louvain
from graphviz import Digraph
from tqdm import tqdm

from kairos_utils import (
    artifact_dir,
    hashgen,
    mean,
    ns_time_to_datetime,
    std,
)

try:
    from kairos_utils import get_db_conn  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    get_db_conn = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Path abstraction dictionary
# --------------------------------------------------------------------------
replace_dic = {
    "/run/shm/": "/run/shm/*",
    "/home/admin/.cache/mozilla/firefox/": "/home/admin/.cache/mozilla/firefox/*",
    "/home/admin/.mozilla/firefox": "/home/admin/.mozilla/firefox*",
    "/data/replay_logdb/": "/data/replay_logdb/*",
    "/home/admin/.local/share/applications/": "/home/admin/.local/share/applications/*",
    "/usr/share/applications/": "/usr/share/applications/*",
    "/lib/x86_64-linux-gnu/": "/lib/x86_64-linux-gnu/*",
    "/proc/": "/proc/*",
    "/stat": "*/stat",
    "/etc/bash_completion.d/": "/etc/bash_completion.d/*",
    "/usr/bin/python2.7": "/usr/bin/python2.7/*",
    "/usr/lib/python2.7": "/usr/lib/python2.7/*",
}


def replace_path_name(path_name: str) -> str:
    for prefix, replacement in replace_dic.items():
        if prefix in path_name:
            return replacement
    return path_name


# --------------------------------------------------------------------------
# Attack windows to visualize (known anomalous time intervals)
# --------------------------------------------------------------------------
DEFAULT_ATTACK_LIST = [
    artifact_dir
    + "/graph_4_6/2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt",
    artifact_dir
    + "/graph_4_6/2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt",
    artifact_dir
    + "/graph_4_6/2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt",
    artifact_dir
    + "/graph_4_6/2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt",
]


# --------------------------------------------------------------------------
# Attack keyword flags (visual emphasis only)
# --------------------------------------------------------------------------
ATTACK_KEYWORDS = {
    "/tmp/vUgefal",
    "vUgefal",
    "/var/log/devc",
    "/etc/passwd",
    "81.49.200.166",
    "61.167.39.128",
    "78.205.235.65",
    "139.123.0.113",
    "'nginx'",
}

SUSPICIOUS_IPS = {
    "81.49.200.166",
    "61.167.39.128",
    "78.205.235.65",
    "139.123.0.113",
}


def attack_edge_flag(msg: str) -> bool:
    return any(keyword in msg for keyword in ATTACK_KEYWORDS)


def is_suspicious_ip(ip: Optional[str]) -> bool:
    return bool(ip and ip in SUSPICIOUS_IPS)


# --------------------------------------------------------------------------
# Dataclasses for enrichment
# --------------------------------------------------------------------------
@dataclass
class TemporalFeatures:
    frequency: int = 0
    mean_interval: float = 0.0
    std_interval: float = 0.0
    is_periodic: bool = False
    burst_score: float = 0.0


@dataclass
class EnrichedNode:
    node_id: str
    msg: str
    entity: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    cmdline: Optional[str] = None
    parent_pid: Optional[int] = None
    user_id: Optional[int] = None
    network_context: Dict[str, Any] = field(default_factory=dict)
    file_metadata: Dict[str, Any] = field(default_factory=dict)
    risk_score: int = 0
    attack_flag: bool = False


# --------------------------------------------------------------------------
# Graph building helpers
# --------------------------------------------------------------------------
def safe_literal_eval(data: str) -> Dict[str, Any]:
    try:
        return ast.literal_eval(data)
    except Exception:
        return {}


def calculate_loss_threshold(losses: List[float], multiplier: float = 1.5) -> float:
    if not losses:
        return 0.0
    mu = mean(losses)
    sigma = std(losses)
    return mu + multiplier * sigma


def try_get_db_conn():
    if callable(get_db_conn):
        try:
            return get_db_conn()
        except Exception as exc:
            print(f"[WARN] DB connection unavailable: {exc}")
    return None


def extract_uuid(entity: Dict[str, Any]) -> Optional[str]:
    for key in ("subject", "file", "netflow"):
        if key in entity and isinstance(entity[key], dict):
            return entity[key].get("uuid")
    return None


def deduce_entity_type(entity: Dict[str, Any]) -> str:
    for key in ("subject", "file", "netflow"):
        if key in entity:
            return key
    return "unknown"


def parse_entity(msg: str) -> Dict[str, Any]:
    entity = safe_literal_eval(msg)
    entity["entity_type"] = deduce_entity_type(entity)
    return entity


def build_attack_graph(attack_paths: Iterable[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    node_frequency: Counter[str] = Counter()
    total_edges = 0

    for path in tqdm(list(attack_paths), desc="Loading anomalous edges"):
        if not path.endswith(".txt"):
            continue
        if not os.path.exists(path):
            print(f"[WARN] Missing anomalous edge file: {path}")
            continue

        with open(path, "r") as fin:
            edges = [safe_literal_eval(line.strip()) for line in fin if line.strip()]

        if not edges:
            continue

        thr = calculate_loss_threshold([edge.get("loss", 0.0) for edge in edges])
        print(
            f"\n[INFO] File: {os.path.basename(path)} | "
            f"edges={len(edges)} | threshold={thr:.4f}"
        )

        for record in edges:
            if record.get("loss", 0.0) <= thr:
                continue

            src_msg = record.get("srcmsg", "")
            dst_msg = record.get("dstmsg", "")
            src_id = str(hashgen(replace_path_name(src_msg)))
            dst_id = str(hashgen(replace_path_name(dst_msg)))

            if not graph.has_node(src_id):
                graph.add_node(
                    src_id,
                    msg=src_msg,
                    msg_display=replace_path_name(src_msg),
                    attack_flag=attack_edge_flag(src_msg),
                    entity=parse_entity(src_msg),
                )
            if not graph.has_node(dst_id):
                graph.add_node(
                    dst_id,
                    msg=dst_msg,
                    msg_display=replace_path_name(dst_msg),
                    attack_flag=attack_edge_flag(dst_msg),
                    entity=parse_entity(dst_msg),
                )

            timestamp_ns = record.get("time")
            timestamp_dt = (
                ns_time_to_datetime(timestamp_ns) if timestamp_ns is not None else None
            )

            graph.add_edge(
                src_id,
                dst_id,
                loss=record.get("loss", 0.0),
                edge_type=record.get("edge_type"),
                time_ns=timestamp_ns,
                time_iso=timestamp_dt.isoformat() if timestamp_dt else None,
                srcmsg=src_msg,
                dstmsg=dst_msg,
            )

            node_frequency[src_id] += 1
            node_frequency[dst_id] += 1
            total_edges += 1

    nx.set_node_attributes(graph, node_frequency, "frequency")
    print(
        f"[INFO] Built graph with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges (filtered from {total_edges} total)."
    )
    return graph


# --------------------------------------------------------------------------
# Enrichment pipeline
# --------------------------------------------------------------------------
class AttackGraphEnricher:
    def __init__(self, conn=None):
        self.conn = conn

    def enrich_graph(self, graph: nx.DiGraph) -> None:
        self._enrich_nodes(graph)
        temporal = TemporalEnricher()
        temporal.analyze(graph)
        context = self._build_context(graph)

        for src, dst, attrs in graph.edges(data=True):
            attrs["importance_score"] = self._calculate_edge_importance(attrs, context)
            attrs["risk_score"] = self._calculate_risk_score(graph.nodes[src], attrs)

    def _enrich_nodes(self, graph: nx.DiGraph) -> None:
        for node_id, attrs in graph.nodes(data=True):
            entity = attrs.get("entity", {})
            metadata = {
                "first_seen": attrs.get("first_seen"),
                "last_seen": attrs.get("last_seen"),
                "risk_score": attrs.get("risk_score", 0),
            }

            if self.conn:
                enriched = self._fetch_db_metadata(entity)
                metadata.update(enriched)

            metadata["risk_score"] = self._calculate_node_risk(attrs, metadata)
            attrs.update(metadata)

    def _fetch_db_metadata(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        if not self.conn:
            return {}
        uuid = extract_uuid(entity)
        if not uuid:
            return {}
        entity_type = entity.get("entity_type")
        if entity_type == "subject":
            return self._get_process_metadata(uuid)
        if entity_type == "file":
            return self._get_file_metadata(uuid)
        if entity_type == "netflow":
            return self._get_network_metadata(uuid)
        return {}

    def _get_process_metadata(self, uuid: str) -> Dict[str, Any]:
        query = """
            SELECT cmdline, ppid, uid, gid, start_time, cwd
            FROM tc_cadet_dataset_db
            WHERE subject_uuid = %s
            ORDER BY start_time ASC
            LIMIT 1
        """
        try:
            with closing(self.conn.cursor()) as cur:  # type: ignore[union-attr]
                cur.execute(query, (uuid,))
                row = cur.fetchone()
            if not row:
                return {}
            cmdline, ppid, uid, gid, start_time, cwd = row
            return {
                "cmdline": cmdline,
                "parent_pid": ppid,
                "user_id": uid,
                "group_id": gid,
                "first_seen": start_time.isoformat() if start_time else None,
                "working_dir": cwd,
            }
        except Exception as exc:
            print(f"[WARN] Process metadata lookup failed for {uuid}: {exc}")
            return {}

    def _get_file_metadata(self, uuid: str) -> Dict[str, Any]:
        query = """
            SELECT path, mode, owner_uid, owner_gid, size_bytes, first_observed, last_observed
            FROM tc_cadet_dataset_db_files
            WHERE file_uuid = %s
            ORDER BY last_observed DESC
            LIMIT 1
        """
        try:
            with closing(self.conn.cursor()) as cur:  # type: ignore[union-attr]
                cur.execute(query, (uuid,))
                row = cur.fetchone()
            if not row:
                return {}
            path, mode, owner_uid, owner_gid, size_bytes, first_obs, last_obs = row
            return {
                "file_metadata": {
                    "path": path,
                    "mode": mode,
                    "owner_uid": owner_uid,
                    "owner_gid": owner_gid,
                    "size_bytes": size_bytes,
                },
                "first_seen": first_obs.isoformat() if first_obs else None,
                "last_seen": last_obs.isoformat() if last_obs else None,
            }
        except Exception as exc:
            print(f"[WARN] File metadata lookup failed for {uuid}: {exc}")
            return {}

    def _get_network_metadata(self, uuid: str) -> Dict[str, Any]:
        query = """
            SELECT local_address, local_port, remote_address, remote_port, protocol, first_seen, last_seen
            FROM tc_cadet_dataset_db_netflows
            WHERE netflow_uuid = %s
            ORDER BY last_seen DESC
            LIMIT 1
        """
        try:
            with closing(self.conn.cursor()) as cur:  # type: ignore[union-attr]
                cur.execute(query, (uuid,))
                row = cur.fetchone()
            if not row:
                return {}
            (
                local_addr,
                local_port,
                remote_addr,
                remote_port,
                protocol,
                first_seen,
                last_seen,
            ) = row
            metadata = {
                "network_context": {
                    "local_address": local_addr,
                    "local_port": local_port,
                    "remote_address": remote_addr,
                    "remote_port": remote_port,
                    "protocol": protocol,
                    "suspicious_ip": is_suspicious_ip(remote_addr),
                },
                "first_seen": first_seen.isoformat() if first_seen else None,
                "last_seen": last_seen.isoformat() if last_seen else None,
            }
            return metadata
        except Exception as exc:
            print(f"[WARN] Netflow metadata lookup failed for {uuid}: {exc}")
            return {}

    def _calculate_node_risk(self, attrs: Dict[str, Any], metadata: Dict[str, Any]) -> int:
        score = 0
        if attrs.get("attack_flag"):
            score += 40
        if attrs.get("frequency", 0) < 3:
            score += 15
        if metadata.get("network_context", {}).get("suspicious_ip"):
            score += 30
        if metadata.get("cmdline") and "nc" in metadata["cmdline"]:
            score += 20
        return min(score, 100)

    def _build_context(self, graph: nx.DiGraph) -> Dict[str, Any]:
        return {
            "edge_type_counts": Counter(
                attrs.get("edge_type") for _, _, attrs in graph.edges(data=True)
            ),
            "high_risk_nodes": {
                node_id
                for node_id, attrs in graph.nodes(data=True)
                if attrs.get("risk_score", 0) >= 60 or attrs.get("attack_flag")
            },
        }

    def _calculate_edge_importance(
        self, attrs: Dict[str, Any], context: Dict[str, Any]
    ) -> int:
        importance = 0.0
        loss = attrs.get("loss", 0.0)
        importance += min(max(loss, 0.0), 1.0) * 40.0

        edge_type = attrs.get("edge_type")
        type_freq = context["edge_type_counts"].get(edge_type, 1)
        rarity_score = 1.0 / np.log(type_freq + 1.5)
        importance += rarity_score * 20.0

        srcmsg = attrs.get("srcmsg", "")
        dstmsg = attrs.get("dstmsg", "")
        if attack_edge_flag(srcmsg) and attack_edge_flag(dstmsg):
            importance += 25.0

        burst_score = attrs.get("temporal_features", {}).get("burst_score", 0.0)
        importance += min(burst_score * 5.0, 15.0)

        return int(min(importance, 100))

    def _calculate_risk_score(
        self, node_attrs: Dict[str, Any], edge_attrs: Dict[str, Any]
    ) -> int:
        score = node_attrs.get("risk_score", 0)
        if edge_attrs.get("importance_score", 0) > 70:
            score += 20
        if attack_edge_flag(edge_attrs.get("dstmsg", "")):
            score += 15
        return min(score, 100)


class TemporalEnricher:
    def analyze(self, graph: nx.DiGraph) -> None:
        edge_timings: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

        for src, dst, attrs in graph.edges(data=True):
            timestamp = attrs.get("time_iso")
            if not timestamp:
                continue
            edge_type = attrs.get("edge_type", "UNKNOWN")
            key = (src, dst, edge_type)
            edge_timings[key].append(self._to_seconds(attrs["time_iso"]))

        for (src, dst, edge_type), timestamps in edge_timings.items():
            if len(timestamps) < 2:
                continue
            timestamps.sort()
            intervals = np.diff(timestamps)
            temporal_features = TemporalFeatures(
                frequency=len(timestamps),
                mean_interval=float(np.mean(intervals)),
                std_interval=float(np.std(intervals)),
                is_periodic=self._is_periodic(intervals),
                burst_score=self._burst_score(timestamps),
            )
            graph.edges[src, dst]["temporal_features"] = temporal_features.__dict__

    @staticmethod
    def _to_seconds(iso_ts: str) -> float:
        return np.datetime64(iso_ts).astype("datetime64[ns]").astype(np.int64) / 1e9

    @staticmethod
    def _is_periodic(intervals: np.ndarray) -> bool:
        if len(intervals) < 3:
            return False
        mean_interval = float(np.mean(intervals))
        if mean_interval == 0:
            return False
        cv = float(np.std(intervals) / mean_interval)
        return cv < 0.3

    @staticmethod
    def _burst_score(timestamps: List[float]) -> float:
        if len(timestamps) < 3:
            return 0.0
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        if mean_interval == 0:
            return 0.0
        return float(np.std(intervals) / mean_interval)


# --------------------------------------------------------------------------
# Community detection and visualization
# --------------------------------------------------------------------------
def run_community_detection(graph: nx.DiGraph) -> Dict[str, int]:
    undirected = graph.to_undirected()
    if undirected.number_of_nodes() == 0:
        return {}
    return community_louvain.best_partition(undirected)


def render_communities(
    graph: nx.DiGraph, partition: Dict[str, int], output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    communities: Dict[int, List[str]] = defaultdict(list)
    for node_id, community_id in partition.items():
        communities[community_id].append(node_id)

    for idx, (community_id, members) in enumerate(communities.items()):
        subgraph = graph.subgraph(members).copy()
        dot = Digraph(name=f"attack_subgraph_{community_id}", format="pdf")
        dot.graph_attr["rankdir"] = "LR"

        for node_id, attrs in subgraph.nodes(data=True):
            risk = attrs.get("risk_score", 0)
            attack_flag = attrs.get("attack_flag", False)
            shape = _node_shape(attrs)
            color = _risk_color(risk, attack_flag)
            label = f"{attrs.get('msg_display', node_id)}\nRISK:{risk}"
            dot.node(node_id, label=label, shape=shape, color=color)

        for src, dst, attrs in subgraph.edges(data=True):
            importance = attrs.get("importance_score", 0)
            edge_color = "red" if importance >= 70 else "dodgerblue2"
            label = f"{attrs.get('edge_type')} (L={attrs.get('loss', 0.0):.3f})"
            dot.edge(src, dst, label=label, color=edge_color)

        outfile = os.path.join(output_dir, f"subgraph_{idx}")
        dot.render(outfile, cleanup=True)
        print(f"[+] Saved subgraph visualization: {outfile}.pdf")


def _node_shape(attrs: Dict[str, Any]) -> str:
    entity_type = attrs.get("entity", {}).get("entity_type")
    if entity_type == "subject":
        return "box"
    if entity_type == "file":
        return "oval"
    if entity_type == "netflow":
        return "diamond"
    return "ellipse"


def _risk_color(risk: int, attack_flag: bool) -> str:
    if attack_flag:
        return "red"
    if risk >= 70:
        return "orangered"
    if risk >= 40:
        return "goldenrod"
    return "steelblue"


# --------------------------------------------------------------------------
# JSON export for UI consumption
# --------------------------------------------------------------------------
def export_json(graph: nx.DiGraph, partition: Dict[str, int], output_path: str) -> None:
    payload = {
        "nodes": [],
        "edges": [],
        "communities": partition,
    }

    for node_id, attrs in graph.nodes(data=True):
        node_payload = {
            "id": node_id,
            "label": attrs.get("msg_display"),
            "entity": attrs.get("entity"),
            "frequency": attrs.get("frequency"),
            "risk_score": attrs.get("risk_score"),
            "attack_flag": attrs.get("attack_flag"),
            "first_seen": attrs.get("first_seen"),
            "last_seen": attrs.get("last_seen"),
            "cmdline": attrs.get("cmdline"),
            "parent_pid": attrs.get("parent_pid"),
            "user_id": attrs.get("user_id"),
            "network_context": attrs.get("network_context"),
            "file_metadata": attrs.get("file_metadata"),
        }
        payload["nodes"].append(node_payload)

    for src, dst, attrs in graph.edges(data=True):
        edge_payload = {
            "source": src,
            "target": dst,
            "edge_type": attrs.get("edge_type"),
            "loss": attrs.get("loss"),
            "time_iso": attrs.get("time_iso"),
            "importance_score": attrs.get("importance_score"),
            "risk_score": attrs.get("risk_score"),
            "temporal_features": attrs.get("temporal_features"),
        }
        payload["edges"].append(edge_payload)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        json.dump(payload, fout, indent=2)
    print(f"[+] Exported enriched graph JSON: {output_path}")


# --------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------
def main(
    attack_paths: Optional[Iterable[str]] = None,
    output_dir: str = f"{artifact_dir}/graph_visual",
    json_output: str = f"{artifact_dir}/graph_visual/attack_investigation.json",
) -> None:
    attack_paths = attack_paths or DEFAULT_ATTACK_LIST
    graph = build_attack_graph(attack_paths)
    if graph.number_of_edges() == 0:
        print("[WARN] No anomalous edges passed the threshold; nothing to visualize.")
        return

    conn = try_get_db_conn()
    enricher = AttackGraphEnricher(conn)
    enricher.enrich_graph(graph)

    partition = run_community_detection(graph)
    render_communities(graph, partition, output_dir)
    export_json(graph, partition, json_output)

    if conn:
        conn.close()


if __name__ == "__main__":
    main()