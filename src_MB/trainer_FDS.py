import os
import shutil
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from es import CMAES
import multiprocessing
import copy
import pickle
from importlib import import_module

from utilities import utils as utils
from models.baselines.FDS_model import Feudal_BL

from config import *


class Envs:
    pass


def simulate(sol_feudal, ep_rewards, ep_timesteps, model_feudal, seed, cfg, envs):
    # set random seeds
    cfg.envs_train = envs.envs_train
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.envs_train.seed(seed)

    t = 0
    done_list = [False for _ in range(cfg.num_envs_train)]
    collect_done = False
    obs_list = cfg.envs_train.reset()

    # Update model parameters using es solution
    model_feudal = utils.update_params_from_layer(model_feudal, cfg.hier_levels, sol_feudal)

    # set initial hierarchical observation (check self.current_h_hier)
    set_obs = torch.from_numpy(obs_list[0]).float().view(cfg.max_num_limbs, -1)
    _ = model_feudal.get_initial_repr(set_obs, cfg.hier_graphs[cfg.envs_train_names[0]])

    while (not collect_done) and (t < cfg.max_episode_steps):
        t += 1
        goal_list = []
        action_list = []

        for i in range(cfg.num_envs_train):
            with torch.no_grad():

                goals, actions = model_feudal(cfg.hier_graphs[cfg.envs_train_names[i]])

                goal_list.append(goals)

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

        # calculate reward for other levels
        reward_envs = []
        for i in range(cfg.num_envs_train):
            if done_list[i]:
                reward_envs.append([0 for _ in range(cfg.hier_levels - 1)])
            else:
                new_env_obs_list = new_obs_list[i]
                new_obs_limbs = new_env_obs_list[:cfg.limb_obs_size * (len(cfg.morph_graphs[cfg.envs_train_names[i]]))]
                old_env_obs_list = obs_list[i]
                old_obs_limbs = old_env_obs_list[:cfg.limb_obs_size * (len(cfg.morph_graphs[cfg.envs_train_names[i]]))]

                diff_obs_limbs = new_obs_limbs - old_obs_limbs
                diff_obs_split = np.array(np.split(diff_obs_limbs, len(cfg.morph_graphs[cfg.envs_train_names[i]])))

                new_obs_split = np.array(np.split(new_obs_limbs, len(cfg.morph_graphs[cfg.envs_train_names[i]])))

                rew = model_feudal.get_reward(goal_list[i], diff_obs_split, torch.from_numpy(new_obs_split),
                                              cfg.hier_graphs[cfg.envs_train_names[i]])
                reward_envs.append(rew)

        reward_envs = list(np.mean(reward_envs, axis=0))

        for lh in range(cfg.hier_levels):
            if lh == (cfg.hier_levels - 1):  # top-level manager has no intrinsic reward
                ep_rewards[lh] += np.mean(reward_list_mgr)
            else:
                ep_rewards[lh] += (reward_envs[lh] + np.mean(reward_list_mgr))

        obs_list = new_obs_list
        collect_done = all(done_list)

    return [ep_rewards, ep_timesteps]


def evaluate_batch(sol_feudal, model_feudal, cfg, envs, seeder=None):
    """Get rewards based on current es parameters"""

    seeds = seeder.next_batch(cfg.pop_size)
    ep_rewards = np.zeros((cfg.pop_size, cfg.hier_levels))
    ep_timesteps = np.zeros((cfg.pop_size, cfg.num_envs_train))
    if cfg.multiprocessing[0]:
        input_args = [(sol_feudal, ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(model_feudal),
                       seeds[pop_id], cfg, envs) for pop_id in range(cfg.pop_size)]
        with multiprocessing.Pool(processes=cfg.multiprocessing[1]) as pool:
            pool_results = pool.starmap(simulate, input_args)
    else:
        pool_results = []
        for pop_id in range(cfg.pop_size):
            pool_results.append(simulate(sol_feudal, ep_rewards[pop_id], ep_timesteps[pop_id],
                                         copy.deepcopy(model_feudal), seeds[pop_id], cfg, envs))
    for idx, result in enumerate(pool_results):
        ep_rewards[idx], ep_timesteps[idx] = result

    return [ep_rewards, ep_timesteps]


class Trainer_FDS:
    def __init__(self, config, save_exp=False):
        self.cfg = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        self.pop_size = self.cfg.pop_size
        self.seeder = utils.Seeder(self.cfg.seed)
        self.seeder_eval = utils.Seeder(self.cfg.seed)

        print("\n" + "-"*20 + " SIMULATION SUMMARY " + "-"*20)
        print("\nRunning feudal baseline")

        if save_exp:
            print("\nExperiment: saved")
        if self.cfg.multiprocessing[0]:
            print("Multiprocessing: enabled with {} cpus".format(self.cfg.multiprocessing[1]))

        if self.cfg.single_optimizer or self.cfg.no_intrinsic_reward:
            raise ValueError("Ablations not supported for feudal baseline")

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
            for name in self.cfg.morphologies:
                shutil.copy('./environments/agents_hierarchy/' + str(self.cfg.hier_levels) + "_"
                            + name.split('_')[0] + '.json', exp_path)
        else:
            exp_name = "EXP_%04d" % self.cfg.exp_id
            exp_path = os.path.join(DATA_DIR, exp_name)
            self.exp_path = exp_path

        # Retrieve MuJoCo XML files for training ========================================
        envs_train_names = []
        self.morph_graphs = dict()
        self.hier_graphs = dict()

        for morphology in self.cfg.morphologies:
            envs_train_names += [name[:-4] for name in os.listdir(XML_DIR) if '.xml' in name and morphology in name]
        for name in envs_train_names:
            graphs = utils.getMultiLevelGraph(name, self.cfg.hier_levels)

            self.morph_graphs[name] = graphs[0]
            self.hier_graphs[name] = graphs[2]
            for graph in self.hier_graphs[name]:
                graph[1] = graph[1] - min(graph[1])

        envs_train_names.sort()

        # sort envs acc to decreasing order of number of limbs (required due to issue in DummyVecEnv)
        order_idx = np.argsort([len(self.morph_graphs[env_name]) for env_name in envs_train_names])[::-1]
        envs_train_names = [envs_train_names[order] for order in order_idx]

        self.num_envs_train = self.cfg.num_envs_train = len(envs_train_names)
        self.envs_train_names = self.cfg.envs_train_names = envs_train_names

        print('Training envs: {}'.format(envs_train_names))
        print('Hierarchical levels: {}'.format(self.cfg.hier_levels))
        print('Population size: {}'.format(self.cfg.pop_size))
        print('Seed: {}\n'.format(self.cfg.seed))
        for name in envs_train_names:
            print(name)
            print('\tMorph graph:\n\t{}'.format(self.morph_graphs[name]))
            print('\tHierarchical graph:\n\t{}\n'.format(self.hier_graphs[name]))

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
        self.cfg.hier_graphs = self.hier_graphs

        # Set up models ================================================

        model_type = import_module('models.' + self.cfg.architecture)
        create_model = getattr(model_type, 'create_model')

        W_1_dim = [self.limb_obs_size, self.cfg.dim_representation]

        rho = [create_model(input_size=self.cfg.dim_representation,
                            output_size=self.cfg.dim_representation,
                            layers=self.cfg.rho_layers) for _ in range(self.cfg.hier_levels - 1)]

        psi = [create_model(input_size=2*self.cfg.dim_representation,
                            output_size=self.limb_obs_size,
                            layers=self.cfg.psi_layers)]

        for lh in range(self.cfg.hier_levels - 2):
            psi.append(create_model(input_size=2*self.cfg.dim_representation,
                                    output_size=self.cfg.dim_representation,
                                    layers=self.cfg.psi_layers))

        mu = create_model(input_size=psi[0].output_size + self.cfg.dim_representation,
                          output_size=self.cfg.mu_out, layers=self.cfg.mu_layers)

        self.model_feudal = Feudal_BL(self.cfg.hier_levels, W_1_dim, rho, psi, mu, seed=self.cfg.seed)

        # Set up evolutions strategy ================================================

        init_params = [None] * self.cfg.hier_levels

        self.es_feudal = []
        config.num_params_model = []

        for lh in range(self.cfg.hier_levels):
            num_params = self.model_feudal.get_num_params_from_layer(lh)
            config.num_params_model.append(num_params)
            cma = CMAES(num_params, init_params[lh], sigma_init=self.cfg.sigma_init, popsize=self.pop_size,
                        seed=self.cfg.seed + lh)
            self.es_feudal.append(cma)

    def train(self):

        total_timesteps = 0

        for episode in range(self.cfg.num_episodes):

            sols_feudal = [cma_lh.ask() for cma_lh in self.es_feudal]       # get parameters from es
            ep_rewards = np.zeros((self.pop_size, self.cfg.hier_levels))

            ep_timesteps = np.zeros((self.pop_size, self.num_envs_train))
            seeds = self.seeder.next_batch(self.pop_size)

            if self.cfg.multiprocessing[0]:
                input_args = [([sols_feudal[lh][pop_id] for lh in range(self.cfg.hier_levels)], ep_rewards[pop_id],
                               ep_timesteps[pop_id], copy.deepcopy(self.model_feudal), seeds[pop_id], self.cfg,
                               self.envs)
                              for pop_id in range(self.pop_size)]
                with multiprocessing.Pool(processes=self.cfg.multiprocessing[1]) as pool:
                    pool_results = pool.starmap(simulate, input_args)
            else:
                pool_results = []
                for pop_id in range(self.pop_size):
                    input_args = ([sols_feudal[lh][pop_id] for lh in range(self.cfg.hier_levels)], ep_rewards[pop_id],
                                  ep_timesteps[pop_id], copy.deepcopy(self.model_feudal), seeds[pop_id], self.cfg,
                                  self.envs)
                    pool_results.append(simulate(*input_args))

            for pop_id, result in enumerate(pool_results):
                ep_rewards[pop_id], ep_timesteps[pop_id] = result
                total_timesteps += ep_timesteps[pop_id].sum()

            # Update parameters using es at the end of episode
            self.es_feudal = utils.run_es_update_params_from_layer(self.cfg.hier_levels, self.es_feudal, ep_rewards.T)

            # Evaluate current params and update best reward
            if episode % self.cfg.eval_episodes == 0:
                sols_feudal = [np.array(es_lh.current_param()).round(4) for es_lh in self.es_feudal]
                reward_eval, ep_timestep_eval = evaluate_batch(sols_feudal, copy.deepcopy(self.model_feudal), self.cfg,
                                                               self.envs, seeder=self.seeder_eval)
                if episode == 0:
                    best_reward_eval = np.mean(reward_eval, axis=0)[-1]
                else:
                    if best_reward_eval < np.mean(reward_eval, axis=0)[-1]:
                        best_reward_eval = np.mean(reward_eval, axis=0)[-1]
                        save_dict = {}
                        save_dict['params_sols'] = [np.array(es.current_param()).round(4) for es in self.es_feudal]

                        with open(os.path.join(self.exp_path, 'best_sols.pickle'), 'wb') as handle:
                            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                reward_eval_level = np.mean(reward_eval, axis=0)
                for lh in range(self.cfg.hier_levels):
                    print("|Episode: {}/{}| Eval reward (level {}): {}".format(episode, self.cfg.num_episodes,
                                                                                  lh+1, reward_eval_level[lh]))
                print("|Episode: {}/{}| Average timesteps: {}\n".format(episode, self.cfg.num_episodes,
                                                                        np.mean(ep_timestep_eval)))
