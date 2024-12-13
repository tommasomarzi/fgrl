import os
import shutil
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from es import CMAES
import multiprocessing
import copy
import pickle
import networkx as nx
from importlib import import_module

from torch_geometric.utils import to_undirected
from utilities import utils as utils
from models.baselines.GNN_model import GNN_BL

from config import *


class Envs:
    pass


def simulate(sol_gnn, ep_rewards, ep_timesteps, model_gnn, seed, cfg, envs):
    # set random seeds
    cfg.envs_train = envs.envs_train
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.envs_train.seed(seed)

    t = 0
    done_list = [False for _ in range(cfg.num_envs_train)]
    collect_done = False
    obs_list = cfg.envs_train.reset()

    model_gnn = utils.update_params_total(model_gnn, sol_gnn)    # Update model parameters using es solution

    while (not collect_done) and (t < cfg.max_episode_steps):
        t += 1
        action_list = []

        for i in range(cfg.num_envs_train):
            with torch.no_grad():
                node_feats = cfg.graph_nfeats[cfg.envs_train_names[i]]          # get node features

                obs = torch.from_numpy(obs_list[i]).float().view(cfg.max_num_limbs, -1)

                actions = model_gnn(cfg.agent_graphs[cfg.envs_train_names[i]], obs, node_feats)

                # add 0-padding to ensure that size is the same for all envs
                actions = np.append(np.array(actions), np.array([0 for _ in range(cfg.max_num_limbs - actions.size)]))
                action_list.append(actions)

        # perform action in the environment
        new_obs_list, reward_list_mgr, curr_done_list, _ = cfg.envs_train.step(action_list)

        # zero out rewards for envs which are done
        reward_list_mgr = [0 if done_list[i] else rew for i, rew in enumerate(reward_list_mgr)]

        # record timesteps for env if done
        for i in range(cfg.num_envs_train):
            if curr_done_list[i]:
                ep_timesteps[i] = t

        # record if each env has ever been 'done'
        done_list = [done_list[i] or curr_done_list[i] for i in range(cfg.num_envs_train)]

        ep_rewards += np.mean(reward_list_mgr)

        obs_list = new_obs_list
        collect_done = all(done_list)

    return [ep_rewards, ep_timesteps]


def evaluate_batch(sol_gnn, model_gnn, cfg, envs, seeder=None):
    """Get rewards based on current es parameters"""

    seeds = seeder.next_batch(cfg.pop_size)
    ep_rewards = np.zeros(cfg.pop_size)
    ep_timesteps = np.zeros((cfg.pop_size, cfg.num_envs_train))
    if cfg.multiprocessing[0]:
        input_args = [(sol_gnn, ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(model_gnn), seeds[pop_id],
                       cfg, envs) for pop_id in range(cfg.pop_size)]
        with multiprocessing.Pool(processes=cfg.multiprocessing[1]) as pool:
            pool_results = pool.starmap(simulate, input_args)
    else:
        pool_results = []
        for pop_id in range(cfg.pop_size):
            pool_results.append(simulate(sol_gnn, ep_rewards[pop_id], ep_timesteps[pop_id],
                                         copy.deepcopy(model_gnn), seeds[pop_id], cfg, envs))
    for idx, result in enumerate(pool_results):
        ep_rewards[idx], ep_timesteps[idx] = result

    return [ep_rewards, ep_timesteps]


class Trainer_GNN:
    def __init__(self, config, save_exp=False):
        self.cfg = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        self.pop_size = self.cfg.pop_size
        self.seeder = utils.Seeder(self.cfg.seed)
        self.seeder_eval = utils.Seeder(self.cfg.seed)

        print("\n" + "-"*20 + " SIMULATION SUMMARY " + "-"*20)
        print("\nRunning gnn baseline")

        if save_exp:
            print("\nExperiment: saved")
        if self.cfg.multiprocessing[0]:
            print("Multiprocessing: enabled with {} cpus".format(self.cfg.multiprocessing[1]))

        if self.cfg.single_optimizer or self.cfg.no_intrinsic_reward:
            raise ValueError("Ablations not supported for GNN baseline")

        # Set up directories ===========================================================
        if save_exp:
            exp_name = "EXP_%04d" % self.cfg.exp_id
            exp_path = os.path.join(DATA_DIR, exp_name)
            self.exp_path = exp_path

            if self.cfg.exp_id == 0:
                os.makedirs(exp_path, exist_ok=True)        # debug mode (overwrite results folder)
            else:
                os.makedirs(exp_path, exist_ok=False)

            shutil.copy('config.py', exp_path)
        else:
            exp_name = "EXP_%04d" % self.cfg.exp_id
            exp_path = os.path.join(DATA_DIR, exp_name)
            self.exp_path = exp_path

        # Retrieve MuJoCo XML files for training ========================================
        envs_train_names = []
        self.morph_graphs = dict()
        self.agent_graphs = dict()

        for morphology in self.cfg.morphologies:
            envs_train_names += [name[:-4] for name in os.listdir(XML_DIR) if '.xml' in name and morphology in name]

        for name in envs_train_names:
            xml_path = os.path.join(XML_DIR, '{}.xml'.format(name))
            morphology = utils.getGraphStructure(xml_path)
            self.morph_graphs[name] = morphology
            directed_graph = torch.tensor(np.array([morphology[1:], np.arange(1, len(morphology))]), dtype=torch.long)
            self.agent_graphs[name] = to_undirected(directed_graph)

        envs_train_names.sort()

        # sort envs acc to decreasing order of number of limbs (required due to issue in DummyVecEnv)
        order_idx = np.argsort([len(self.morph_graphs[env_name]) for env_name in envs_train_names])[::-1]
        envs_train_names = [envs_train_names[order] for order in order_idx]

        self.num_envs_train = self.cfg.num_envs_train = len(envs_train_names)
        self.envs_train_names = self.cfg.envs_train_names = envs_train_names

        print('Training envs: {}'.format(envs_train_names))
        print('Message-passing rounds: {}'.format(self.cfg.mp_rounds))
        print('Population size: {}'.format(self.cfg.pop_size))
        print('Seed: {}\n'.format(self.cfg.seed))
        for name in envs_train_names:
            print(name)
            print('\tMorph graph:\n\t{}'.format(self.morph_graphs[name]))
            print('\tAgent graph:\n\t{}'.format(self.agent_graphs[name]))

        # Set up training env ================================================
        self.limb_obs_size, self.max_action = utils.registerEnvs(envs_train_names, self.cfg.max_episode_steps,
                                                                 self.cfg.custom_xml)
        self.cfg.limb_obs_size = self.limb_obs_size

        self.cfg.max_num_limbs = self.max_num_limbs = max([len(self.morph_graphs[env_name])
                                                           for env_name in envs_train_names])
        # create vectorized training env
        obs_max_len = max([len(self.morph_graphs[env_name]) for env_name in envs_train_names]) * self.limb_obs_size
        envs_train = [utils.makeEnvWrapper(name, obs_max_len, self.cfg.seed) for name in envs_train_names]

        self.envs_train = DummyVecEnv(envs_train)  # vectorized env (necessary for multiprocessing)
        self.envs = Envs
        self.envs.envs_train = self.envs_train

        # determine the maximum number of children in all the training envs
        self.max_children = utils.findMaxChildren(envs_train_names, self.morph_graphs)

        self.cfg.morph_graphs = self.morph_graphs
        self.cfg.agent_graphs = self.agent_graphs

        # Get graph node features (distance from torso)
        self.graph_nfeats = dict()
        if self.cfg.enable_features:
            node_feats_dim = 1  # adding only one feature (distance from torso)
            for g_name, g_struct in self.morph_graphs.items():
                edge_list = np.array([g_struct[1:], np.arange(1, len(g_struct))]).T.tolist()
                g = nx.DiGraph(edge_list)
                node_feats = np.array([nx.shortest_path_length(g, 0, n_id) for n_id in range(len(g_struct))]).reshape(-1, 1)
                self.graph_nfeats[g_name] = torch.tensor(node_feats, dtype=torch.long).view(-1, node_feats_dim)
        else:
            node_feats_dim = 0
            for env in envs_train_names:
                self.graph_nfeats[env] = None

        self.cfg.graph_nfeats = self.graph_nfeats

        # Set up models ================================================

        model_type = import_module('models.' + self.cfg.architecture)
        create_model = getattr(model_type, 'create_model')

        W_1_dim = [self.limb_obs_size + node_feats_dim, self.cfg.dim_representation]

        phi_1 = create_model(input_size=2*self.cfg.dim_representation,
                             output_size=self.cfg.dim_representation,
                             layers=self.cfg.phi_1_layers)

        mu = create_model(input_size=self.cfg.dim_representation,
                          output_size=self.cfg.mu_out, layers=self.cfg.mu_layers)

        self.model_gnn = GNN_BL(self.cfg.mp_rounds, W_1_dim, phi_1, mu, seed=self.cfg.seed)

        # Set up evolutions strategy ================================================

        init_params = None

        config.num_params_model = []

        num_params = self.model_gnn.get_model_num_params()
        config.num_params_model.append(num_params)
        self.es_gnn = CMAES(num_params, init_params, sigma_init=self.cfg.sigma_init, popsize=self.pop_size,
                            seed=self.cfg.seed)

    def train(self):

        total_timesteps = 0

        for episode in range(self.cfg.num_episodes):

            sols_gnn = self.es_gnn.ask()     # get parameters from es
            ep_rewards = np.zeros(self.pop_size)

            ep_timesteps = np.zeros((self.pop_size, self.num_envs_train))
            seeds = self.seeder.next_batch(self.pop_size)

            if self.cfg.multiprocessing[0]:
                input_args = [(sols_gnn[pop_id], ep_rewards[pop_id], ep_timesteps[pop_id],
                               copy.deepcopy(self.model_gnn), seeds[pop_id], self.cfg, self.envs)
                              for pop_id in range(self.pop_size)]
                with multiprocessing.Pool(processes=self.cfg.multiprocessing[1]) as pool:
                    pool_results = pool.starmap(simulate, input_args)
            else:
                pool_results = []
                for pop_id in range(self.pop_size):
                    input_args = (sols_gnn[pop_id], ep_rewards[pop_id], ep_timesteps[pop_id],
                                  copy.deepcopy(self.model_gnn), seeds[pop_id], self.cfg, self.envs)
                    pool_results.append(simulate(*input_args))

            for pop_id, result in enumerate(pool_results):
                ep_rewards[pop_id], ep_timesteps[pop_id] = result
                total_timesteps += ep_timesteps[pop_id].sum()

            # Update parameters using es at the end of episode
            self.es_gnn = utils.run_es_update_params_total(self.es_gnn, ep_rewards)
            self.es_gnn = self.es_gnn[0]      # run_es_update_params_total() returns a list

            # Evaluate current params and update best reward
            if episode % self.cfg.eval_episodes == 0:
                sols_gnn = np.array(self.es_gnn.current_param()).round(4)
                reward_eval, ep_timestep_eval = evaluate_batch(sols_gnn, copy.deepcopy(self.model_gnn), self.cfg,
                                                               self.envs, seeder=self.seeder_eval)
                if episode == 0:
                    best_reward_eval = np.mean(reward_eval)
                else:
                    if best_reward_eval < np.mean(reward_eval):
                        best_reward_eval = np.mean(reward_eval)
                        save_dict = {}
                        save_dict['params_sols'] = np.array(self.es_gnn.current_param()).round(4)

                        with open(os.path.join(self.exp_path, 'best_sols.pickle'), 'wb') as handle:
                            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print("|Episode: {}/{}| Eval reward: {}".format(episode, self.cfg.num_episodes,
                                                                    np.mean(reward_eval)))
                print("|Episode: {}/{}| Average timesteps: {}\n".format(episode, self.cfg.num_episodes,
                                                                        np.mean(ep_timestep_eval)))