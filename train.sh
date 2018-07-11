#!/bin/bash

# Remove all __pycache__ files
rm -rf {rl/__pycache__,rl/agents/a2c/__pycache__,rl/networks/__pycache__}
export SLACK_API_TOKEN=''

# Run
#python run.py move_to_beacon                 --save_iters 500 --nchw --envs 16 --map MoveToBeacon
#python run.py collect_mineral_shards         --save_iters 500 --nchw --envs 16 --map CollectMineralShards
#python run.py find_and_defeat_zerglings      --save_iters 500 --nchw --envs 16 --map FindAndDefeatZerglings
python run.py defeat_roaches                   --debug --save_iters 500 --nchw --envs 16 --map DefeatRoaches
#python run.py defeat_zerglings_and_banelings  --lr 0.00095 --save_iters 500 --nchw --envs 16 --map DefeatZerglingsAndBanelings
#python run.py collect_minerals_and_gas       --save_iters 500 --nchw --envs 16 --map CollectMineralsAndGas
#python run.py build_marines                  --save_iters 500 --nchw --envs 16 --map BuildMarines
