# StateMorph

A full re-implementation for [StateMorph](https://link.springer.com/chapter/10.1007/978-3-319-68456-7_4) in python. Taking advantage of (Dask)[https://www.dask.org/] and its distributed library, together with python ecosystem.

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
    client, n_state, init_temp=init_temp, final_temp=final_temp, alpha=alpha)
client.upload_file('state_morph.zip')
trainer.load_raw_corpus(wordlist)
model = trainer.train(n_epoch)
client.close()
cluster.close()
# Save models
StateMorphIO().write_binary_model_file(model, model_path+'.bin', no_corpus=True)
StateMorphIO().write_segmented_file(model, model_path+'.txt')
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

from state_morph import StateMorphIO
from state_morph import StateMorphTrainer
from dask.distributed import Client, LocalCluster, SSHCluster
import argparse
import os
import shutil


if __name__ ==  '__main__':
    argparser = argparse.ArgumentParser(description='A script to train StateMorph')
    argparser.add_argument('--model_path', type=str, required=True, help='Save path for the model')
    argparser.add_argument('--wordlist', type=str, required=True, help='The wordlist file')
    argparser.add_argument('--init_temp', type=float, required=True, help='Initial temperature')
    argparser.add_argument('--final_temp', type=float, required=True, help='Final temperature')
    argparser.add_argument('--alpha', type=float, required=True, help='Alpha for simulated annealing')
    argparser.add_argument('--n_epoch', type=int, required=True, help='Upper bound of the number of epochs')
    argparser.add_argument('--n_state', type=int, required=True, help='Number of states')
    
    argparser.add_argument('--multi_node',default=False, action='store_true', help='Multi node or not')
    argparser.add_argument('--n_worker', type=int, default=1,  help='Number of workers per node')
    argparser.add_argument('--threads_per_worker', type=int, default=1,  help='Number of threads per worker')
    args = argparser.parse_args()
    
    
    
    
    worker_options = {'n_workers': args.n_worker}
    if not args.multi_node:
        worker_options['threads_per_worker'] = args.threads_per_worker
        cluster = LocalCluster(**worker_options)
    else:
        master_node = os.environ['SLURMD_NODENAME']
        stream = os.popen('scontrol show hostname')
        worker_nodes = stream.read().splitlines()
        worker_options['nthreads'] = args.threads_per_worker
        cluster = SSHCluster([master_node] + worker_nodes, 
                             connect_options={"known_hosts": None, 'username': 'XXXX'},
                             worker_options=worker_options,
                             scheduler_options={"port": 0, "dashboard_address": ":8797"})
    
    shutil.make_archive('state_morph', 'zip', './dummy')
    
    client = Client(cluster)
    trainer = StateMorphTrainer(
        client, args.n_state, init_temp=args.init_temp, final_temp=args.final_temp, alpha=args.alpha)
    client.upload_file('state_morph.zip')
    trainer.load_raw_corpus(os.path.abspath(args.wordlist))
    model = trainer.train(args.n_epoch)
    client.close()
    cluster.close()
    os.remove('state_morph.zip')
    wordlist_name = os.path.splitext(os.path.basename(args.wordlist))[0]
    model_name = '{}_{}_{}_{}_{}'.format(wordlist_name, args.n_state, args.init_temp, args.final_temp, args.alpha)
    model_path = os.path.join(os.path.abspath(args.model_path), model_name)
    
    StateMorphIO().write_binary_model_file(model, model_path+'.bin', no_corpus=True)
    StateMorphIO().write_segmented_file(model, model_path+'.txt')

```