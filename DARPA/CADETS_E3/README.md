## Project Overview

The DARPA CADETS E3 project implements a temporal graph neural network for cybersecurity anomaly detection. It processes provenance graph data from the DARPA TC (Transparent Computing) program to detect malicious activities.


## I. Environment Setup

### Prerequisites
- Ubuntu 24+
- PostgreSQL 16+ 
- CUDA-capable GPU (CUDA 12.8)
- Miniconda/Anaconda

### Step 1: Create Python (Conda) Environment and install dependencies
Set up the Python environment (3.10):
```bash
../../env_setup_310.sh
```

### Step 2: Set up PostgreSQL
Run the PostgreSQL setup script from the root directory:
```bash
../../psql_setup.sh
```

### Step 3: Activate Conda Environment
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate python310
```

### Step 4: Set up Database for CADETS E3
Create the specific database for CADETS E3:
```bash
# Create database schema and tables (tables (SQL commands from DARPA/settings/database.md for CADETS E3)
./database_setup.sh

```


## II. Data Acquisition and Preparation

### Step 1: Download Raw Data
```bash
# Make the data download script executable and run it
./get_data.sh
```

### Step 2: Configure Data Paths
Edit ../src/config.py to set the correct paths:
```python
# Update the raw_dir path to point to your downloaded data
raw_dir = "/path/to/your/downloaded/cadets_e3_data/"
```

### Step 3: Decode Binary Files (if needed)
If you have binary files that need decoding:
```bash
# Edit the script to set correct paths, then run:
./decode_binfiles.sh
```


## III. Data Processing Pipeline

### Step 1: Prepare Artifacts Directory
```bash
make prepare
```

### Step 2: Create Database and Parse Raw Data
```bash
make create_data
```
This script (create_database.py) will:
- Parse JSON log files
- Extract nodes (subjects, files, network flows)
- Store processed data in PostgreSQL tables

### Step 3: Generate Graph Embeddings
```bash
make embed_graphs
```
This runs `embedding.py` to:
- Create temporal graph snapshots
- Generate node embeddings
- Save vectorized graphs for training


## 4. Model Training and Testing

### Step 1: Train the Model
```bash
make train
```
This runs train.py which:
- Loads the temporal graph data
- Trains a temporal graph neural network (TGN)
- Saves the trained model

### Step 2: Test the Model
```bash
make test
```
This runs test.py to:
- Load the trained model
- Perform link prediction on test data
- Generate reconstruction scores

### Step 3: Anomaly Detection
```bash
make anomalous_queue
```
This constructs anomalous event queues for evaluation.

### Step 4: Evaluation
```bash
make evaluation
```
This runs the evaluation metrics and generates results.

### Step 5: Attack Investigation
```bash
make attack_investigation
```
This performs attack path investigation analysis.

## 5. Complete Pipeline

To run the entire pipeline:
```bash
# Full preprocessing pipeline
make preprocess

# Deep learning pipeline  
make deep_graph_learning

# Anomaly detection pipeline
make anomaly_detection

# Or run everything at once
make pipeline
```


## 6. Configuration Details

### Key Configuration Parameters (in config.py):

- **Data Paths**: Update `raw_dir` to your data location
- **Database Settings**: Configure PostgreSQL connection details
- **Model Parameters**: 
  - `node_embedding_dim = 16`
  - `node_state_dim = 100` 
  - `time_window_size = 60000000000 * 15` (15 minutes)
- **Training Parameters**:
  - `BATCH = 1024`
  - `lr = 0.00005`
  - `epoch_num = 50`

### Expected Data Format:
The system expects JSON files with names like:
- `ta1-cadets-e3-official.json`
- `ta1-cadets-e3-official.json.1`
- etc.

## 7. Troubleshooting

1. **PostgreSQL Connection Issues**: Check the troubleshooting section in environment-settings.md
2. **CUDA Issues**: Ensure PyTorch is installed with CUDA support
3. **Memory Issues**: Adjust batch size in config.py if you encounter OOM errors
4. **Path Issues**: Ensure all paths in config.py are absolute paths


## 8. Output Artifacts

The pipeline will create:
- `./artifact/graphs/`: Vectorized temporal graphs
- `./artifact/models/`: Trained model files
- `./artifact/test_re/`: Test results and reconstruction scores
- `./artifact/vis_re/`: Visualization results

This system implements a sophisticated temporal graph neural network for cybersecurity anomaly detection, processing provenance graphs to identify malicious activities in system call traces.
