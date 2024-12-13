import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_es_update_params_total(model, es, rew):
    """
    Run the ES using given reward list and update the model parameters using the new ES solution
    :param model: FGRL model
    :param es: evolution strategy object
    :param rew: list of environment rewards
    :return: PyTorch model, list of ES objects
    """

    es.tell(rew)

    return model, [es]


def update_params_total(model, solution):
    """
    Update the model params for each layer using the given solution
    :param model: PyTorch model
    :param solution: list of ES solutions to be used as new Parameters
    :return: PyTorch model
    """

    model_params = solution         # best historical solution
    model_params = np.array(model_params).round(4)

    funcs = model.get_model_funcs()
    new_state_dict = funcs.state_dict()

    param_idx = 0
    for p, v in new_state_dict.items():
        params_len = np.prod(v.shape)
        new_params = torch.Tensor(model_params[param_idx: param_idx + params_len].round(4).reshape(v.shape))
        new_state_dict[p] = new_params
        param_idx += params_len

    funcs.load_state_dict(new_state_dict)

    return model


def run_es_update_params_from_layer(model, hier_levels, es, rew_list):
    """
    Run the ES using given reward list and update the model parameters using the new ES solution
    :param model: FGRL model
    :param hier_levels: number of hierarchical levels
    :param es: list of evolution strategy objects
    :param rew_list: list of rewards
    :return: PyTorch model, list of ES objects
    """

    for lh in range(hier_levels):
        es[lh].tell(rew_list[lh])

    return model, es


def update_params_from_layer(model, hier_levels, solution):
    """
    Update the model params for each layer using the given solution
    :param model: PyTorch model
    :param hier_levels: number of hierarchical levels
    :param solution: list of ES solutions to be used as new Parameters
    :return: PyTorch model
    """

    for lh in range(hier_levels):
        model_params = solution[lh]         # best historical solution
        model_params = np.array(model_params).round(4)

        funcs_lh = model.get_funcs_from_layer(lh)
        new_state_dict = funcs_lh.state_dict()

        param_idx = 0
        for p, v in new_state_dict.items():
            params_len = np.prod(v.shape)
            new_params = torch.Tensor(model_params[param_idx: param_idx + params_len].round(4).reshape(v.shape))
            new_state_dict[p] = new_params
            param_idx += params_len

        funcs_lh.load_state_dict(new_state_dict)

    return model


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31-1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result

    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result
