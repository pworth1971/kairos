################################################################################
# Makefile for CADETS E3 Data Processing and Deep Graph Learning Pipeline
#
# This Makefile defines the full workflow for building the environment,
# preparing data, generating embeddings, training Temporal Graph Networks (TGNs),
# detecting anomalies, and investigating security events.
#
# Usage examples:
#   make setup_conda          # Set up Python environment
#   make setup_postgres       # Install and configure PostgreSQL
#   make init_database        # Initialize database schema
#   make preprocess           # Create dataset and embeddings
#   make deep_graph_learning  # Train and test the model
#   make anomaly_detection    # Detect and evaluate anomalies
#   make pipeline             # Run the full end-to-end process
################################################################################


# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------

# 1. Setup Conda environment for Python 3.10
setup_conda:
	../../../setup/env_setup_310.sh

# 2. Install and configure PostgreSQL database (version 16)
setup_postgres:
	../../../setup/psql_setup.sh

# 3. Initialize the database schema and tables
init_database:
	./database_setup.sh

# 4. Prepare artifact directories for storing generated data/models
prepare:
	mkdir -p ../artifact/


# ------------------------------------------------------------------------------
# Data ingestion and preprocessing
# ------------------------------------------------------------------------------

# 5. Create and populate the PostgreSQL database with parsed CADETS E3 data
create_data:
	python ../src/create_database.py

# 6. Generate node and edge embeddings (using feature hashing / message encoding)
embed_graphs:
	python ../src/embedding.py


# ------------------------------------------------------------------------------
# Model training and testing
# ------------------------------------------------------------------------------

# 7. Train the Temporal Graph Network (TGN) on the generated embeddings
train:
	python ../src/train.py

# 8. Evaluate the trained model on held-out datasets
test:
	python ../src/test.py


# ------------------------------------------------------------------------------
# Anomaly detection and evaluation
# ------------------------------------------------------------------------------

# 9. Construct anomalous queues using node loss profiles over time windows
anomalous_queue:
	python ../src/anomalous_queue_construction.py

# 10. Evaluate anomaly detection performance (precision, recall, etc.)
evaluation:
	python ../src/evaluation.py


# ------------------------------------------------------------------------------
# Security event investigation
# ------------------------------------------------------------------------------

# 11. Analyze detected attacks and generate investigation reports
attack_investigation:
	python ../src/attack_investigation.py


# ------------------------------------------------------------------------------
# High-level pipeline shortcuts
# ------------------------------------------------------------------------------

# === Environment initialization ===
# Sets up conda, PostgreSQL, initializes DB, and prepares directories
initialize: setup_conda setup_postgres init_database prepare

# === Data preprocessing ===
# Parses CADETS dataset and creates graph embeddings
preprocess: create_data embed_graphs

# === Deep Graph Learning (TGN Training) ===
# Trains and evaluates the TGN model
deep_graph_learning: train test

# === Anomaly Detection ===
# Constructs anomaly queues and evaluates detection performance
anomaly_detection: anomalous_queue evaluation

# === Full End-to-End Pipeline ===
# Runs all phases: preprocessing → training → anomaly detection → attack analysis
pipeline: preprocess deep_graph_learning anomaly_detection attack_investigation


