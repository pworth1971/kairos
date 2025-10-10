##########################################################################################
# 
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
#
##########################################################################################

import logging
from kairos_utils import *
from model import *

import torch
from torch_geometric.data.storage import GlobalStorage

# allowlist GlobalStorage so torch.load can unpickle it
torch.serialization.add_safe_globals([GlobalStorage])



# --------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------- 
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# --------------------------------------------------------------------------
# Early Stopping Utility
# --------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when validation loss doesn't improve after 'patience' epochs."""
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStop] No improvement for {self.counter}/{self.patience} epochs (best={self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStop] Triggered at epoch {epoch}. Best epoch: {self.best_epoch}")


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
    
    logger.info(f"-- train() ---")

    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    
    #for batch in train_data.seq_batches(batch_size=BATCH):
    
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
        
        optimizer.zero_grad()

        #src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Prepare node neighborhood for message passing
        n_id = torch.cat([src, pos_dst]).unique()
        
        # Retrieve neighbors
        n_id, edge_index, e_id = neighbor_loader(n_id)

        # Create assoc on the correct device
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Forward pass through memory and GNN
        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
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

    logger.info(f"-- load_train_data() ---")

    graph_4_2 = torch.load(graphs_dir + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]


def init_models_orig(node_feat_size):
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


# --------------------------------------------------------------------------------------------
#   Supports paralleization (simplest form) 
#   Multi-GPU (DataParallel) Version
# --------------------------------------------------------------------------------------------

def init_models(node_feat_size):

    logger.info(f"-- init_model(node_feat_size:({node_feat_size}) ---")

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

    # ✅ Multi-GPU: wrap models in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel.")
        memory = torch.nn.DataParallel(memory)
        gnn = torch.nn.DataParallel(gnn)
        link_pred = torch.nn.DataParallel(link_pred)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=lr, eps=eps, weight_decay=weight_decay
    )

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)
    return memory, gnn, link_pred, optimizer, neighbor_loader


# --------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    start_time = time.time()
    logger.info("Start logging.")
    logger.info(f"Detected device: {device}")

    # Load data for training
    train_data = load_train_data()
    train_data = [move_to_device(g, device) for g in train_data]
    
    # Data statistics
    total_events = sum(g.src.size(0) for g in train_data)
    feature_dim = train_data[0].msg.size(-1)
    logger.info(f"Loaded {len(train_data)} temporal graphs.")
    logger.info(f"Total events: {total_events:,}")
    logger.info(f"Feature dimension: {feature_dim}")

    # Initialize the models and the optimizer
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    # Training loop
    epoch_losses = []
    early_stopper = EarlyStopping(patience=5, min_delta=1e-3, verbose=True)

    # train the model
    for epoch in tqdm(range(1, epoch_num+1)):
    
        epoch_start = time.time()
        epoch_loss = 0.0
    
        for g in train_data:
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader
            )
            epoch_loss += loss
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_data)
        epoch_losses.append(avg_loss)
        logger.info(f"Epoch {epoch:02d}/{epoch_num}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        print(f"[INFO] Epoch {epoch:02d}/{epoch_num} completed — Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # --- Early Stopping Check ---
        early_stopper(avg_loss, epoch)
        if early_stopper.early_stop:
            print(f"[EarlyStop] Stopping training early at epoch {epoch}")
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        
    total_time = time.time() - start_time


    # Save trained model
        
    """
    model = [memory, gnn, link_pred, neighbor_loader]
    os.system(f"mkdir -p {models_dir}")
    torch.save(model, f"{models_dir}/models.pt")
    """

    model_state = {
        'memory': memory.module.state_dict() if isinstance(memory, torch.nn.DataParallel) else memory.state_dict(),
        'gnn': gnn.module.state_dict() if isinstance(gnn, torch.nn.DataParallel) else gnn.state_dict(),
        'link_pred': link_pred.module.state_dict() if isinstance(link_pred, torch.nn.DataParallel) else link_pred.state_dict(),
    }

    os.makedirs(models_dir, exist_ok=True)
    
    torch.save(model_state, f"{models_dir}/models.pt")

    # ----------------------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------------------
    avg_epoch_time = total_time / epoch_num
    final_loss = epoch_losses[-1]
    min_loss = min(epoch_losses)
    max_loss = max(epoch_losses)

    summary = f"""
================= Training Summary =================
Device:           {device}
GPUs Used:        {torch.cuda.device_count() if torch.cuda.is_available() else 0}
Epochs:           {epoch_num}
Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} min)
Avg Epoch Time:   {avg_epoch_time:.2f} seconds
Final Loss:       {final_loss:.4f}
Best Loss:        {min_loss:.4f}
Worst Loss:       {max_loss:.4f}
Graphs Used:      {len(train_data)}
Total Events:     {total_events:,}
Feature Dim:      {feature_dim}
Model Saved To:   {models_dir}/models.pt
====================================================
"""
    print(summary)
    logger.info(summary)