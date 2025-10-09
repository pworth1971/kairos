# Environment settings

## Prerequisites
We use the following settings to run the experiments reported in the paper:
1. OS Version: 5.19.0-46-generic #47~22.04.1-Ubuntu
2. Anaconda: 23.3.1
3. PostgresSQL: Version 15.3, Ubuntu 15.3-1.pgdg22.04+1 ([installation guide](https://www.cherryservers.com/blog/how-to-install-and-setup-postgresql-server-on-ubuntu-20-04))
4. GraphViz: 2.43.0 
5. GPU (Driver Version: 530.41.03): CUDA Version 12.1

## Python Libraries
Install the following libraries, 
or use our [`requirements.txt`](requirements.txt):
```commandline
conda create -n kairos python=3.9
conda activate kairos
# Note: using "pip install psycopg2" to install may fail
conda install psycopg2
conda install tqdm
# We encountered a problem in feature hashing functions with version 1.2.2
pip install scikit-learn==1.2.0
pip install networkx==2.8.7
pip install xxhash==3.2.0
pip install graphviz==0.20.1

# PyTorch GPU version
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.0.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

```

## Troubleshooting

**Issue**: When running `psycopg2.connect()`, I received this error:
```
OperationalError: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  Peer authentication failed for user "postgres".
```

**Solution**: Follow the solution in [this](https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge) Stack Overflow post.

**Issue**: When running `psycopg2.connect()`, I received this error:
```
OperationalError: could not connect to server: No such file or directory the server running locally and accepting connections on Unix domain socket "/XXX/.s.PGSQL.5432"?
```

**Solution**:
* Check if `postgres` is running. If not, start it, re-run the code, and see if the problem still exists.
* If the problem still exists when `postgres` is running, identify the location of the file `.s.PGSQL.5432`. 
Then set the `host` parameter in `psycopg2.connect()` to be `/the/location/of/the/file/`. The problem should be fixed then.
