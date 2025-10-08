# Copilot Instructions for KAIROS

## Project Overview
KAIROS is a temporal graph neural network (TGN) system for cybersecurity anomaly detection using whole-system provenance data. It processes system call traces from DARPA Transparent Computing datasets to detect intrusions through graph reconstruction.

## Architecture & Data Flow

### Core Pipeline (CADETS_E3 example)
1. **Data Ingestion**: Raw JSON logs → PostgreSQL (`create_database.py`)
2. **Graph Construction**: DB events → temporal graph snapshots (`embedding.py`)
3. **Training**: TGN model training on link prediction (`train.py`)
4. **Detection**: Reconstruction scores → anomaly detection (`test.py`, `anomalous_queue_construction.py`)
5. **Investigation**: Attack path analysis (`attack_investigation.py`)

### Key Components
- **`config.py`**: Central configuration (paths, DB settings, model params)
- **`kairos_utils.py`**: Core utilities (time handling, DB connections, graph ops)
- **`model.py`**: TGN architecture (GraphAttentionEmbedding, LinkPredictor)
- **Makefile**: Pipeline orchestration with logical groupings

## Dataset Structure
- **DARPA/CADETS_E3/**: Full pipeline implementation (primary reference)
- **DARPA/{THEIA,CLEARSCOPE,OpTC}_E{3,5}/**: Jupyter notebooks for other datasets
- **StreamSpot/**: Alternative graph-based detection approach

## Development Workflows

### Environment Setup
```bash
# Use project scripts for consistent environment
./env_setup_39.sh  # Creates python39 conda env with dependencies
./postgres_setup.sh  # Sets up PostgreSQL 15+
```

### Running Experiments
```bash
cd DARPA/CADETS_E3
make pipeline  # Full end-to-end execution
# OR step-by-step:
make preprocess  # prepare + create_database + embed_graphs
make deep_graph_learning  # train + test
make anomaly_detection  # anomalous_queue + evaluation
```

### Configuration Patterns
- **Always update `raw_dir` in `config.py`** to absolute path of dataset
- **Database connection**: Check `host` parameter for PostgreSQL socket issues
- **Memory management**: Adjust `BATCH` size in config for OOM errors
- **Time windows**: Default 15-min windows (`time_window_size = 60000000000 * 15`)

## Code Conventions

### Database Integration
- Use `psycopg2` with cursor contexts: `with conn.cursor() as cur:`
- Nanosecond timestamps throughout: `ns_time_to_datetime()` for conversion
- Table naming: `{dataset}_dataset_db` (e.g., `tc_cadet_dataset_db`)

### Graph Processing
- **Node types**: SUBJECT (processes), FILE, NETFLOW with hashed IDs
- **Edge semantics**: 7 core event types in `include_edge_type` list
- **Reversible edges**: `EVENT_ACCEPT`, `EVENT_RECVFROM`, `EVENT_RECVMSG`
- **Feature encoding**: Hierarchical path/IP hashing in `embedding.py`

### Model Architecture
- **Temporal encoding**: Relative timestamps between events
- **Attention mechanism**: TransformerConv with 8 heads → 1 head
- **Link prediction**: Concatenated src/dst embeddings → MLP classifier

## File Organization Patterns
- **`artifact/`**: All generated outputs (graphs, models, results)
  - `graphs/`: Vectorized temporal graphs
  - `models/`: Trained TGN checkpoints  
  - `test_re/`: Reconstruction scores
  - `vis_re/`: Attack investigation visualizations
- **`Data/`**: Raw dataset files (`.json`, `.bin`)
- **`Ground_Truth/`**: Attack labels for evaluation

## Testing & Evaluation
- **Pre-trained models**: Available on Google Drive for quick testing
- **Evaluation metrics**: AUC, precision, recall in `evaluation.py`
- **Memory requirements**: 64GB+ RAM for full datasets (StreamSpot)
- **GPU acceleration**: CUDA support required for efficient training

## Common Pitfalls
- **Path issues**: Always use absolute paths in `config.py`
- **PostgreSQL authentication**: May need `host='/var/run/postgresql/'` 
- **Dataset variations**: Each DARPA dataset has different schemas/formats
- **Memory constraints**: Large graphs require batch processing and careful memory management
- **Timestamp precision**: Nanosecond precision critical for temporal ordering

## Integration Points
- **Multi-dataset support**: Notebooks in each dataset folder handle schema differences
- **Provenance graphs**: Standard PROV-DM model with dataset-specific adaptations
- **External tools**: GraphViz for visualization, PostgreSQL for storage
- **Evaluation frameworks**: Custom metrics aligned with DARPA TC evaluation