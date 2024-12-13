import argparse
from torch import set_num_threads
import importlib

from trainer_FGNN import Trainer
from trainer_GNN import Trainer_GNN
from trainer_FDS import Trainer_FDS
from trainer_DS import Trainer_DS
from trainer_MLP import Trainer_MLP

from config import *


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save_exp", action='store_true',
                    help="Save the experiment")

    args = vars(ap.parse_args())
    save_exp = args['save_exp']

    config = Config()

    set_num_threads(1)

    if not config.use_baseline[0]:
        trainer = Trainer(config, save_exp=save_exp)
    else:
        if config.use_baseline[1] == 'GNN':
            trainer = Trainer_GNN(config, save_exp=save_exp)
        elif config.use_baseline[1] == 'FDS':
            trainer = Trainer_FDS(config, save_exp=save_exp)
        elif config.use_baseline[1] == 'DS':
            trainer = Trainer_DS(config, save_exp=save_exp)
        elif config.use_baseline[1] == 'MLP':
            trainer = Trainer_MLP(config, save_exp=save_exp)
        else:
            raise ValueError("Baseline not implemented")

    trainer.train()
