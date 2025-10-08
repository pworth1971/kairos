##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging

from kairos_utils import *
from config import *
from model import *

# --------------------------------------------------------------------------
# Device detection (Apple Silicon, CUDA, or CPU)
# --------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[INFO] Using Apple Silicon Metal (MPS) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[INFO] Using CUDA GPU device.")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU device (no GPU acceleration available).")

# set default device for PyTorch    
torch.set_default_device(device)


# --------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------- 
LOG_DIR="./log/"
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_DIR + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# --------------------------------------------------------------------------
# Training function
# --------------------------------------------------------------------------
def train(train_data,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader
          ):
    
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    
    #for batch in train_data.seq_batches(batch_size=BATCH):
    
    # ---------------- fix for torch_geometric.data.TemporalData object -------------------
    num_events = train_data.src.size(0)
    for start in range(0, num_events, BATCH):
        end = min(start + BATCH, num_events)

        # Create a mini-batch manually (as a dict)
        batch = {
            'src': train_data.src[start:end],
            'dst': train_data.dst[start:end],
            't': train_data.t[start:end],
            'msg': train_data.msg[start:end],
        }
        
        src = batch['src']
        pos_dst = batch['dst']
        t = batch['t']
        msg = batch['msg']
        
        #--------------------------------- end fix -----------------------------

        optimizer.zero_grad()

        #src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Prepare node neighborhood for message passing
        n_id = torch.cat([src, pos_dst]).unique().to(device)
        
        # Retrieve neighbors (LastNeighborLoader may return CPU tensors)
        n_id, edge_index, e_id = neighbor_loader(n_id)

        # Create assoc on the correct device
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Forward pass through memory and GNN
        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id.to("cpu"))
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # Compute the target tensor
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        # Loss and optimization
        loss = criterion(y_pred, y_true)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        
        total_loss += float(loss) * len(src)
#        total_loss += float(loss) * batch.num_events
        
    #return total_loss / train_data.num_events
    return total_loss / num_events


# --------------------------------------------------------------------------
# Data and model initialization
# --------------------------------------------------------------------------
def load_train_data():
    graph_4_2 = torch.load(graphs_dir + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]

def init_models(node_feat_size):
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        node_state_dim,
        time_dim,
        message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=lr, eps=eps, weight_decay=weight_decay)

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    return memory, gnn, link_pred, optimizer, neighbor_loader


def move_to_device(obj, device):
    """Recursively move all torch tensors (or lists/dicts) to the target device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [move_to_device(x, device) for x in obj]
    else:
        return obj



MODELS_DIR="./models/"

# --------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    logger.info("Start logging.")
    logger.info(f"Detected device: {device}")

    # Load data for training
    train_data = load_train_data()
    train_data = [move_to_device(g, device) for g in train_data]
    
    # Initialize the models and the optimizer
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    # train the model
    for epoch in tqdm(range(1, epoch_num+1)):
        for g in train_data:
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # Save the trained model
    model = [memory, gnn, link_pred, neighbor_loader]

    os.system(f"mkdir -p {MODELS_DIR}")
    torch.save(model, f"{MODELS_DIR}/models.pt")
