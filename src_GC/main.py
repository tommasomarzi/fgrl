from torch import set_num_threads
import json
import argparse
from trainer_FGNN import Trainer
from trainer_GNN import Trainer_GNN
from trainer_FDS import Trainer_FDS
from trainer_DS import Trainer_DS
from trainer_MLP import Trainer_MLP
import importlib

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save_exp", action='store_true',
                    help="Save the experiment")

    args = vars(ap.parse_args())
    save_exp = args['save_exp']

    config_path = "config.json"
    with open(config_path) as cf:
        config = json.load(cf)

    num_cpus = 1
    set_num_threads(num_cpus)

    if not config["simulation"]["use_baseline"][0]:
        trainer = Trainer(config, save_exp=save_exp)
    else:
        if config["simulation"]["use_baseline"][1] == 'GNN':
            trainer = Trainer_GNN(config, save_exp=save_exp)
        elif config["simulation"]["use_baseline"][1] == 'FDS':
            trainer = Trainer_FDS(config, save_exp=save_exp)
        elif config["simulation"]["use_baseline"][1] == 'DS':
            trainer = Trainer_DS(config, save_exp=save_exp)
        elif config["simulation"]["use_baseline"][1] == 'MLP':
            trainer = Trainer_MLP(config, save_exp=save_exp)
        else:
            raise ValueError("Baseline not implemented")
    trainer.train()