
##########################################################################################
#
#   Temporal Graph Reconstruction and Evaluation
#   Adapted from PyTorch Geometric TGN example: 
#   https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
#
##########################################################################################


import logging

from kairos_utils import *
from model import *


from torch_geometric.data.storage import GlobalStorage
import torch.serialization

# Allow GlobalStorage class to be deserialized safely
torch.serialization.add_safe_globals([GlobalStorage])


# --------------------------- LOGGING SETUP ---------------------------

logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + 'reconstruction.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@torch.no_grad()            # Disable gradient tracking — inference only

def test(inference_data,
          memory,
          gnn,
          link_pred,
          neighbor_loader,
          nodeid2msg,
          path
          ):
    """
    Run model inference on temporal graph data and log per-time-window reconstruction losses.
    """
    
    # Create output directory for the test run
    if not os.path.exists(path):
        os.mkdir(path)
        
    memory.eval()
    gnn.eval()
    link_pred.eval()

    # Reset temporal memory and neighbor cache
    memory.reset_state()                            # Start with a fresh memory.
    neighbor_loader.reset_state()                   # Start with an empty graph.

    # Containers for metrics
    time_with_loss = {}                 # {time_window: {loss, nodes_count, edges_count, time}}
    total_loss = 0
    edge_list = []                      # Store per-edge losses and metadata

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0

    # Initialize first time window
    start_time = inference_data.t[0]
    event_count = 0
    pos_o = []

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    #
    # Loop through temporal batches of events
    #
    num_events = inference_data.src.size(0)

    logger.info(f"[START] Inference loop over {num_events:,} events (batch={BATCH})")
    print(f"[INFO] Starting reconstruction on {num_events:,} events...")

    # Initialize progress bar
    progress_bar = tqdm(range(0, num_events, BATCH), desc="Reconstructing", ncols=100)

    for start in progress_bar:
        end = min(start + BATCH, num_events)

        #src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # create manual batch slice
        src = inference_data.src[start:end]
        pos_dst = inference_data.dst[start:end]
        t = inference_data.t[start:end]
        msg = inference_data.msg[start:end]
        num_events_batch = end - start

        # Track unique nodes across all edges
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += BATCH

        # Build neighborhood subgraph for message passing
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Compute embeddings (forward pass)
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        # Predictions (pos_out) — compare to ground truth (y_true)
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []

        # Extract ground-truth relation type from the message tensor
        # Each msg = [src_emb | relation_emb | dst_emb]
        # We slice the middle section (relation_emb) and find where the label == 1
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)
        
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        # Compute loss
        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # Update temporal memory (stateful)
        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # compute the loss for each edge
        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        # Record metadata for every edge
        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][node_embedding_dim:-node_embedding_dim], 1)
            edge_type = rel2id[edgeindex]
            loss = each_edge_loss[i]

            temp_dic = {}
            temp_dic['loss'] = float(loss)
            temp_dic['srcnode'] = srcnode
            temp_dic['dstnode'] = dstnode
            temp_dic['srcmsg'] = srcmsg
            temp_dic['dstmsg'] = dstmsg
            temp_dic['edge_type'] = edge_type
            temp_dic['time'] = t_var

            edge_list.append(temp_dic)

        # Write results at end of a time window
        event_count += len(batch.src)
        if t[-1] > start_time + time_window_size:
            # Here is a checkpoint, which records all edge losses in the current time window
            time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval] = {'loss': loss,
                                             'nodes_count': len(unique_nodes),
                                             'total_edges': total_edges,
                                             'costed_time': (end - start)}

            # Write per-edge logs sorted by loss
            log = open(path + "/" + time_interval + ".txt", 'w')
            for e in edge_list:
                loss += e['loss']
            loss = loss / event_count
            logger.info(
                f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
            # Rank the results based on edge losses
            edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            
            logger.info(
                f"[CHECKPOINT] Time: {time_interval} | Avg Loss: {avg_loss:.4f} | "
                f"Nodes: {len(unique_nodes)} | Edges: {total_edges:,} | "
                f"Duration: {(end_clock - start_clock):.2f}s"
            )
            print(f"[INFO] {time_interval} — Avg Loss={avg_loss:.4f}, Edges={total_edges:,}")

            # Reset counters for next window
            event_count = 0
            total_loss = 0
            start_time = t[-1]
            log.close()
            edge_list.clear()

    logger.info("[DONE] Inference complete for all batches.")
    progress_bar.close()
    return time_with_loss




def load_data():
    """Load training (for initialization) and testing graphs."""

    # graph_4_3 - graph_4_5 will be used to initialize node IDF scores.
    graph_4_3 = torch.load(graphs_dir+"/graph_4_3.TemporalData.simple", weights_only=False).to(device=device)
    graph_4_4 = torch.load(graphs_dir+"/graph_4_4.TemporalData.simple", weights_only=False).to(device=device)
    graph_4_5 = torch.load(graphs_dir+"/graph_4_5.TemporalData.simple", weights_only=False).to(device=device)

    # Testing set
    graph_4_6 = torch.load(graphs_dir+"/graph_4_6.TemporalData.simple", weights_only=False).to(device=device)
    graph_4_7 = torch.load(graphs_dir+"/graph_4_7.TemporalData.simple", weights_only=False).to(device=device)

    return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]




if __name__ == "__main__":
    logger.info("Start logging.")

    # load the map between nodeID and node labels
    #cur, _ = init_database_connection()
    cur, _ = init_database_connection2()
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Load data
    graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = load_data()

    # load trained model
    memory, gnn, link_pred, neighbor_loader = torch.load(
                                                f"{models_dir}/models.pt",
                                                weights_only=False, 
                                                map_location=device)

    # Reconstruct the edges in each day
    test(inference_data=graph_4_3,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_3")

    test(inference_data=graph_4_4,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_4")

    test(inference_data=graph_4_5,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_5")

    test(inference_data=graph_4_6,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_6")

    test(inference_data=graph_4_7,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_7")
