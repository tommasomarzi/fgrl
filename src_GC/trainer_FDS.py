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
from env.clustering_env import Graph_Clustering

from utilities import utils as utils
from models.baselines.FDS_model import Feudal_BL

DATA_DIR = './results'


class Envs:
    pass


def simulate(sol_feudal, ep_rewards, ep_timesteps, model_feudal, seed, cfg, envs):
    # set random seeds
    cfg["envs_train"] = envs.envs_train
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg["envs_train"].seed(seed)

    t = 0
    done = False
    is_solved = int(False)
    obs = cfg["envs_train"].reset()
    obs = obs[0]

    # Update model parameters using es solution
    model_feudal = utils.update_params_from_layer(model_feudal, cfg["policy"]["hier_levels"], sol_feudal)

    # set initial hierarchical observation (check self.current_h_hier)
    set_obs = torch.from_numpy(obs).float()
    _ = model_feudal.get_initial_repr(set_obs)

    # set goal step counter to zero
    model_feudal.goal_step_counter = 0

    while (not done) and (t < cfg["environment"]["max_episode_steps"]):
        t += 1

        with torch.no_grad():
            goals, actions = model_feudal()
            actions = actions.argmax(axis=1)

        # perform action in the environment
        new_obs, reward, done, infos = cfg["envs_train"].step([actions])
        new_obs = new_obs[0]
        reward = reward[0]
        done = done[0]
        infos = infos[0]

        # record timesteps for env if done
        if done:
            # DummyVecEnv resets when done, so set the new_obs with the terminal observation
            new_obs = cfg["envs_train"].buf_infos[0]['terminal_observation']
            ep_timesteps = t
            is_solved = int(infos['solved_task'])
        
        # calculate reward for other levels
        # diff_obs = new_obs - obs
        rew_levels = model_feudal.get_reward(torch.from_numpy(new_obs),
                                             par_goals=[goals, new_obs])

        reward_timestep = []
        for idx, lh in enumerate(range(cfg["policy"]["hier_levels"])[::-1]):     # cumulate rewards from top to bottom
            if idx == 0:
                reward_timestep.append(reward)
            else:
                reward_timestep.append(rew_levels[lh]/cfg["environment"]["max_episode_steps"] +
                                       reward_timestep[0])

        for lh, rew_lh in enumerate(reward_timestep[::-1]):         # add rewards from bottom to top
            ep_rewards[lh] += rew_lh


        obs = new_obs

    return [ep_rewards, ep_timesteps, is_solved]


def evaluate_batch(sol_feudal, model_feudal, cfg, envs, seeder=None):
    """Get rewards based on current es parameters"""

    seeds = seeder.next_batch(cfg["optimizer"]["pop_size"])
    ep_rewards = np.zeros((cfg["optimizer"]["pop_size"], cfg["policy"]["hier_levels"]))
    ep_timesteps = np.zeros((cfg["optimizer"]["pop_size"], 1))
    ep_solved = np.zeros((cfg["optimizer"]["pop_size"], 1))
    if cfg["utilities"]["multiprocessing"][0]:
        input_args = [(sol_feudal, ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(model_feudal), seeds[pop_id],
                       cfg, envs) for pop_id in range(cfg["optimizer"]["pop_size"])]
        with multiprocessing.Pool(processes=cfg["utilities"]["multiprocessing"][1]) as pool:
            pool_results = pool.starmap(simulate, input_args)
    else:
        pool_results = []
        for pop_id in range(cfg["optimizer"]["pop_size"]):
            pool_results.append(simulate(sol_feudal, ep_rewards[pop_id], ep_timesteps[pop_id],
                                         copy.deepcopy(model_feudal), seeds[pop_id], cfg, envs))
    for idx, result in enumerate(pool_results):
        ep_rewards[idx], ep_timesteps[idx], ep_solved[idx] = result

    return [ep_rewards, ep_timesteps, ep_solved]


class Trainer_Feudal:
    def __init__(self, config, save_exp=True):
        self.cfg = config

        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        if self.cfg["simulation"]["seed"]["custom_seed"]:
            self.seed = self.cfg["simulation"]["seed"]["custom_seed_value"]
        else:
            self.seed = self.cfg["simulation"]["exp_id"]

        self.pop_size = self.cfg["optimizer"]["pop_size"]
        self.seeder = utils.Seeder(self.seed)
        self.seeder_eval = utils.Seeder(self.seed)

        print("\n" + "-"*20 + " SIMULATION SUMMARY " + "-"*20)

        if save_exp:
            print("\nExperiment: saved")
        if self.cfg["utilities"]["multiprocessing"][0]:
            print("Multiprocessing: enabled with {} cpus".format(self.cfg["utilities"]["multiprocessing"][1]))

        # Set up directories ===========================================================
        if save_exp:
            exp_name = "EXP_%04d" % self.cfg["simulation"]["exp_id"]
            exp_path = os.path.join(DATA_DIR, exp_name)
            self.exp_path = exp_path

            if self.cfg["simulation"]["exp_id"] == 0:
                os.makedirs(exp_path, exist_ok=True)        # debug mode (overwrite results folder)
            else:
                os.makedirs(exp_path, exist_ok=False)

            shutil.copy('config.json', exp_path)

        print('Hierarchical levels: {}'.format(self.cfg["policy"]["hier_levels"]))
        print('Population size: {}'.format(self.cfg["optimizer"]["pop_size"]))
        print('Seed: {}'.format(self.seed))

        # create vectorized training env
        # envs_train = [utils.makeEnvWrapper(name, obs_max_len, self.cfg.seed) for name in envs_train_names]
        env = Graph_Clustering(self.cfg["environment"])

        self.envs_train = DummyVecEnv([lambda:env])  # vectorized env
        self.envs = Envs
        self.envs.envs_train = self.envs_train
        # self.envs = env
        self.node_feats_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.nvec[0]

        # Set up models ================================================

        hyperparams = self.cfg["policy"]["hyperparameters"]

        model_type = import_module('models.' + self.cfg["policy"]["architecture"])
        create_model = getattr(model_type, 'create_model')

        W_1_dim = [self.node_feats_dim, hyperparams["initial_repr_out"][0]]

        rho = [create_model(input_size=hyperparams["initial_repr_out"][lh],
                            output_size=hyperparams["initial_repr_out"][lh+1],
                            layers=hyperparams["internal_dim"]) for lh in range(self.cfg["policy"]["hier_levels"] - 1)]

        psi = [create_model(input_size=self.cfg["environment"]["n_connected_comp"] + hyperparams["initial_repr_out"][0],
                            output_size=self.cfg["environment"]["n_connected_comp"],
                            layers=hyperparams["internal_dim"],
                            last_activation="softmax")]
        for lh in range(self.cfg["policy"]["hier_levels"] - 2):
            psi.append(create_model(input_size= hyperparams["initial_repr_out"][lh+1]
                                                + hyperparams["initial_repr_out"][lh+2],
                                    output_size=self.cfg["environment"]["n_connected_comp"],
                                    layers=hyperparams["internal_dim"],
                                    last_activation="softmax"))

        mu = create_model(input_size=psi[0].output_size + hyperparams["initial_repr_out"][0],
                          output_size=self.action_dim, layers=hyperparams["internal_dim"],
                          last_activation="softmax")

        if self.cfg["policy"]["goal_scale"][0]:
            goal_scale = self.cfg["policy"]["goal_scale"][1]
        else:
            goal_scale = None

        self.model_feudal = Feudal_BL(self.cfg["policy"]["hier_levels"], W_1_dim, rho, psi, mu=mu,
                                      seed=self.seed, goal_scale=goal_scale, cfg_env=self.cfg["environment"])

        # Set up evolutions strategy ================================================

        init_params = [None] * self.cfg["policy"]["hier_levels"]
        init_sigma = [self.cfg["optimizer"]["sigma_init"]] * self.cfg["policy"]["hier_levels"]

        self.es_feudal = []
        config["num_params_model"] = []

        for lh in range(self.cfg["policy"]["hier_levels"]):
            num_params = self.model_feudal.get_num_params_from_layer(lh)
            config["num_params_model"].append(num_params)
            cma = CMAES(num_params, init_params[lh], sigma_init=init_sigma[lh], popsize=self.pop_size,
                        seed=self.seed + lh)
            self.es_feudal.append(cma)

    def train(self):
        list_solved = [0]*20

        total_timesteps = 0

        for episode in range(self.cfg["simulation"]["num_episodes"]):

            sols_feudal = [cma_lh.ask() for cma_lh in self.es_feudal]       # get parameters from es
            ep_rewards = np.zeros((self.pop_size, self.cfg["policy"]["hier_levels"]))
            ep_timesteps = np.zeros((self.pop_size, 1))
            ep_solved = np.zeros((self.pop_size, 1))
            seeds = self.seeder.next_batch(self.pop_size)

            if self.cfg["utilities"]["multiprocessing"][0]:
                input_args = [([sols_feudal[lh][pop_id] for lh in range(self.cfg["policy"]["hier_levels"])],
                               ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(self.model_feudal), seeds[pop_id],
                               self.cfg, self.envs)
                              for pop_id in range(self.pop_size)]
                with multiprocessing.Pool(processes=self.cfg["utilities"]["multiprocessing"][1]) as pool:
                    pool_results = pool.starmap(simulate, input_args)
            else:
                pool_results = []
                for pop_id in range(self.pop_size):
                    input_args = ([sols_feudal[lh][pop_id] for lh in range(self.cfg["policy"]["hier_levels"])],
                                  ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(self.model_feudal),
                                  seeds[pop_id], self.cfg, self.envs)
                    pool_results.append(simulate(*input_args))

            for pop_id, result in enumerate(pool_results):
                ep_rewards[pop_id], ep_timesteps[pop_id], ep_solved[pop_id] = result
                total_timesteps += ep_timesteps[pop_id].sum()

            # Update parameters using es at the end of episode
            self.model_feudal, self.es_feudal = utils.run_es_update_params_from_layer(self.model_feudal,
                                                                                      self.cfg["policy"]["hier_levels"],
                                                                                      self.es_feudal, ep_rewards.T)

            # Evaluate current params and update best reward
            if episode % self.cfg["simulation"]["eval_episodes_freq"] == 0:
                sols_feudal = [np.array(es_lh.current_param()).round(4) for es_lh in self.es_feudal]
                reward_eval, ep_timestep_eval, solved_eval = evaluate_batch(sols_feudal, copy.deepcopy(self.model_feudal),
                                                                            self.cfg, self.envs,
                                                                            seeder=self.seeder_eval)
                if self.cfg["simulation"]["exp_id"] != 0:
                    save_dict = {}
                    save_dict['params_sols'] = [np.array(es.current_param()).round(4) for es in self.es_feudal]
                    save_dict['sigma'] = [np.array(es.current_sigma()).round(4) for es in self.es_feudal]
                    save_dict['rms_sigma'] = [np.array(es.rms_stdev()).round(4) for es in self.es_feudal]
                    with open(os.path.join(self.exp_path, 'last_sols.pickle'), 'wb') as handle:
                        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if episode == 0:
                    best_reward_eval = np.mean(reward_eval, axis=0)[-1]
                else:
                    if best_reward_eval < np.mean(reward_eval, axis=0)[-1]:
                        best_reward_eval = np.mean(reward_eval, axis=0)[-1]
                        if self.cfg["simulation"]["exp_id"] != 0:
                            with open(os.path.join(self.exp_path, 'best_sols.pickle'), 'wb') as handle:
                                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                reward_eval_level = np.mean(reward_eval, axis=0)
                for lh in range(self.cfg["policy"]["hier_levels"]):
                    print("|Episode: {}| Eval reward (level {}): {}".format(episode, lh, reward_eval_level[lh]))
                print("|Episode: {}| Percentage of solved : {}| Average timesteps: {}".format(episode, np.mean(solved_eval),np.mean(ep_timestep_eval)))

                list_solved.pop(0)
                list_solved.append(np.mean(solved_eval))
                if np.mean(list_solved) > 0.95:
                    break
