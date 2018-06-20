#!/bin/bash

# Remove all __pycache__ files
rm -rf {rl/__pycache__,rl/agents/a2c/__pycache__,rl/networks/__pycache__}

# Run
python run.py collect_mineral_shards --save_iters 500 --nchw --envs 16 --map CollectMineralShards
