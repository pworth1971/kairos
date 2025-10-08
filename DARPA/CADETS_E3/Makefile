prepare:
	mkdir -p ./artifact/

create_database:
	python create_database.py

embed_graphs:
	python ../src/embedding.py

train:
	python ../src/train.py

test:
	python ../src/test.py

anomalous_queue:
	python ../src/anomalous_queue_construction.py

evaluation:
	python ../src/evaluation.py

attack_investigation:
	python ../src/attack_investigation.py

preprocess: prepare create_database embed_graphs

deep_graph_learning: train test

anomaly_detection: anomalous_queue evaluation

pipeline: preprocess deep_graph_learning anomaly_detection attack_investigation

