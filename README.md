# StateMorph

A re-implementation for [StateMorph](https://link.springer.com/chapter/10.1007/978-3-319-68456-7_4) in python. Taking advantage of [Dask](https://www.dask.org/) and its distributed library, together with python ecosystem.

## Training on local

```python

from state_morph import StateMorphIO
from state_morph import StateMorphTrainer
from dask.distributed import Client, LocalCluster
import os
import shutil

cluster = LocalCluster(n_workers=n_workers) 
client = Client(cluster)
trainer = StateMorphTrainer(
    client, 
    n_workers,
    n_state,                    # Number of expected state, including start state and ending state
    model_path,                 # Save path for the model
    model_name,                 # Model name
    init_temp=init_temp,        # Initial temperature for simulated annealing
    final_temp=final_temp,      # Final temperature for simulated annealing
    alpha=alpha,                # Alpha for simulated annealing
    num_prefix=n_prefix,        # Number of prefixes
    num_suffix=n_suffix,        # Number of suffix
    affix_lbound=affix_lbound,  # Lower bound of the freq of affix states
    stem_ubound=stem_ubound,    # Upper bound of the freq of stem states
    bulk_prob=bulk_prob         # Probability of bulk deregistering morpheme
)

trainer.load_raw_corpus(wordlist_file)
model = trainer.train(max_epoch)
client.close()
cluster.close()

```

## Training with cluster

```python

from dask.distributed import Client, SSHCluster


cluster = SSHCluster([master_node] + worker_nodes, 
                     connect_options={"known_hosts": None, 'username': 'XXXX'},
                     worker_options={'n_workers': n_worker, 'memory_limit': None}, 
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}) 

```

NB: Dask share memory with worker evenly. Yet *Reduce* may require extra memory than *Map*. Reset `memory_limit` to meet the requirement.

## Fine tune an existing model

```python
trainer = StateMorphTrainer(...)
trainer.load_checkpoint( BIN_MODEL_FILE )

```

## Running model

```python
from state_morph import StateMorphIO
model = StateMorphIO().load_model_from_binary_file(bin_file_path)
# model = StateMorphIO().load_model_from_text_files(
#     num_state, num_prefix, num_suffix, segmented_file)
model.segment('XXX')

```