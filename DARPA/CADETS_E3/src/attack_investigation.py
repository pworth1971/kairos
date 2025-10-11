##########################################################################################
# Attack Graph Visualization and Community Detection
#
# Purpose:
#   - Parse anomalous edge logs (from reconstruction results) for known attack windows
#   - Build directed graphs representing suspicious node interactions
#   - Perform community detection (Louvain clustering) to identify coherent subgraphs
#   - Visualize subgraphs using Graphviz with color-coded nodes and edges
#
# Inputs:
#   - Anomalous edge logs (graph_4_6/DATE_RANGE.txt)
# Outputs:
#   - Visual subgraph PDFs under artifact_dir/graph_visual/
##########################################################################################
# 


import os

from graphviz import Digraph
import networkx as nx
import datetime
import community.community_louvain as community_louvain
from tqdm import tqdm

from kairos_utils import *


# --------------------------------------------------------------------------
# Path abstraction dictionary
# --------------------------------------------------------------------------
# This dictionary replaces verbose or system-specific file paths
# with generalized tokens to simplify visualization and anonymize data.
# 
# Some common path abstraction for visualization
#
replace_dic = {
    '/run/shm/': '/run/shm/*',
    '/home/admin/.cache/mozilla/firefox/': '/home/admin/.cache/mozilla/firefox/*',
    '/home/admin/.mozilla/firefox': '/home/admin/.mozilla/firefox*',
    '/data/replay_logdb/': '/data/replay_logdb/*',
    '/home/admin/.local/share/applications/': '/home/admin/.local/share/applications/*',
    '/usr/share/applications/': '/usr/share/applications/*',
    '/lib/x86_64-linux-gnu/': '/lib/x86_64-linux-gnu/*',
    '/proc/': '/proc/*',
    '/stat': '*/stat',
    '/etc/bash_completion.d/': '/etc/bash_completion.d/*',
    '/usr/bin/python2.7': '/usr/bin/python2.7/*',
    '/usr/lib/python2.7': '/usr/lib/python2.7/*',
}

def replace_path_name(path_name):
    """Replace system file paths with abstracted tokens for cleaner graph display."""

    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]

    return path_name


# --------------------------------------------------------------------------
# Attack windows to visualize (known anomalous time intervals)
# Users should manually put the detected anomalous time windows here
# --------------------------------------------------------------------------
attack_list = [
    artifact_dir+'/graph_4_6/2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
    artifact_dir+'/graph_4_6/2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
    artifact_dir+'/graph_4_6/2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
    artifact_dir+'/graph_4_6/2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
]

# --------------------------------------------------------------------------
# Step 1. Load anomalous edges and filter high-loss edges
# --------------------------------------------------------------------------
original_edges_count = 0
graphs = []
gg = nx.DiGraph()
count = 0
for path in tqdm(attack_list):
    if ".txt" in path:
        line_count = 0
        node_set = set()

         # Temporary directed graph for this time window
        tempg = nx.DiGraph()
        f = open(path, "r")
        edge_list = []
        for line in f:
            count += 1
            l = line.strip()
            jdata = eval(l)
            edge_list.append(jdata)

        # Sort edges by reconstruction loss (descending)
        edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)
        original_edges_count += len(edge_list)

        # Compute loss threshold (mean + 1.5σ) for anomaly filtering
        loss_list = []
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean = mean(loss_list)
        loss_std = std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr = loss_mean + 1.5 * loss_std

        print(f"\n[INFO] File: {os.path.basename(path)}")
        print(f"  mean={loss_mean:.4f}, std={loss_std:.4f}, threshold={thr:.4f}")

        # Add high-loss (anomalous) edges to both local and global graphs
        for e in edge_list:
            if e['loss'] > thr:
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),
                               str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))), str(hashgen(replace_path_name(e['dstmsg']))),
                            loss=e['loss'], srcmsg=e['srcmsg'], dstmsg=e['dstmsg'], edge_type=e['edge_type'],
                            time=e['time'])


# --------------------------------------------------------------------------
# Step 2. Apply community detection (Louvain modularity optimization)
# --------------------------------------------------------------------------
# Converts the directed graph to undirected for community clustering.
partition = community_louvain.best_partition(gg.to_undirected())

#
# Initialize subgraphs per community ID
# Generate the candidate subgraphs based on community discovery results
#
communities = {}        
max_partition = 0
for i in partition:
    if partition[i] > max_partition:
        max_partition = partition[i]
for i in range(max_partition + 1):
    communities[i] = nx.DiGraph()

# Assign edges to community subgraphs
for e in gg.edges:
    communities[partition[e[0]]].add_edge(e[0], e[1])
    communities[partition[e[1]]].add_edge(e[0], e[1])


# --------------------------------------------------------------------------
# Step 3. Define attack-related keywords (for coloring only)
# 
# Define the attack nodes. They are **only be used to plot the colors of attack nodes and edges**.
# They won't change the detection results.
#
# --------------------------------------------------------------------------
def attack_edge_flag(msg):
    """
    Identify whether a node/edge is part of a known attack indicator.
    This is purely for visualization (red highlights) and does not alter logic.
    """
    attack_nodes = [
        '/tmp/vUgefal',
        'vUgefal',
        '/var/log/devc',
        '/etc/passwd',
        '81.49.200.166',
        '61.167.39.128',
        '78.205.235.65',
        '139.123.0.113',
        "'nginx'",
    ]
    flag = False
    for i in attack_nodes:
        if i in msg:
            flag = True
    return flag

# --------------------------------------------------------------------------
# Step 4. Visualize each community as a PDF subgraph using Graphviz
# 
# Plot and render candidate subgraph
# --------------------------------------------------------------------------
os.system(f"mkdir -p {artifact_dir}/graph_visual/")
graph_index = 0
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'

    for e in communities[c].edges:
        try:
            temp_edge = gg.edges[e]
            srcnode = e['srcnode']
            dstnode = e['dstnode']
        except:
            pass

        if True:
            # set shape and color for a given node based on 
            # its type and threat level (source node)
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape = 'box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape = 'oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape = 'diamond'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color = 'red'
            else:
                src_node_color = 'blue'
            dot.node(name=str(hashgen(replace_path_name(temp_edge['srcmsg']))), label=str(
                replace_path_name(temp_edge['srcmsg']) + str(
                    partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), color=src_node_color,
                     shape=src_shape)

            # Create nodes in the graph (destination node)
            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape = 'box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape = 'oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape = 'diamond'
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color = 'red'
            else:
                dst_node_color = 'blue'
            dot.node(name=str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=str(
                replace_path_name(temp_edge['dstmsg']) + str(
                    partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), color=dst_node_color,
                     shape=dst_shape)

            # Create edges in Graphviz graph
            if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
                edge_color = 'red'
            else:
                edge_color = 'blue'
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=temp_edge['edge_type'],
                     color=edge_color)

    # Save each subgraph as a PDF
    out_path = f'{artifact_dir}/graph_visual/subgraph_'
    dot.render( out_path + str(graph_index), view=False)
    print(f"[+] Saved subgraph visualization: {out_path}.pdf")

    graph_index += 1



