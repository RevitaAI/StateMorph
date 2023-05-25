# StateMorph

A re-implementation for [StateMorph](https://link.springer.com/chapter/10.1007/978-3-319-68456-7_4) in python. Taking advantage of (Dask)[https://www.dask.org/] and its distributed library, together with python ecosystem.

## Training on local

```python

from state_morph import StateMorphIO
from state_morph import StateMorphTrainer
from dask.distributed import Client, LocalCluster
import os
import shutil

cluster = LocalCluster(n_workers=n_workers) 
client = Client(cluster)
trainer = StateMorphTrainer(client, n_state, model_path, model_name, 
                            init_temp=init_temp, final_temp=final_temp, alpha=alpha)

trainer.load_raw_corpus(wordlist)
model = trainer.train(n_epoch)
client.close()
cluster.close()

```

## Training with cluster

```python

from dask.distributed import Client, SSHCluster


cluster = SSHCluster([master_node] + worker_nodes, 
                     connect_options={"known_hosts": None, 'username': 'XXXX'},
                     worker_options={'n_workers': n_worker},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}) 

```
## Running model
(TBD)

```python

```