#!/bin/bash

# Remove all __pycache__ files
rm -rf {rl/__pycache__,rl/agents/a2c/__pycache__,rl/networks/__pycache__}

# Run
python run.py collect_mineral_shards --eval --envs 16 --map CollectMineralShards
