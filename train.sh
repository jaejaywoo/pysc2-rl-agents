#!/bin/bash

# Remove all __pycache__ files
rm -rf {rl/__pycache__,rl/agents/a2c/__pycache__,rl/networks/__pycache__}
export SLACK_API_TOKEN=''

# Run
#python run.py move_to_beacon                 --lr 0.0007 --save_iters 500 --nchw --envs 16 --map MoveToBeacon
#python run.py collect_mineral_shards         --lr 0.0007 --save_iters 500 --nchw --envs 16 --map CollectMineralShards
#python run.py find_and_defeat_zerglings      --lr 0.0007 --save_iters 500 --nchw --envs 16 --map FindAndDefeatZerglings
#python run.py defeat_roaches                 --lr 0.0007 --save_iters 500 --nchw --envs 16 --map DefeatRoaches
#python run.py defeat_zerglings_and_banelings --lr 0.0007 --save_iters 500 --nchw --envs 16 --map DefeatZerglingsAndBanelings
#python run.py collect_minerals_and_gas       --lr 0.0007 --save_iters 500 --nchw --envs 16 --map CollectMineralsAndGas
