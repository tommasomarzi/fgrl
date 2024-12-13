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
from models.baselines.GNN_model import GNN_BL

DATA_DIR = './results'


class Envs:
    pass


def simulate(sol_GNN, ep_rewards, ep_timesteps, model_GNN, seed, cfg, envs):
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
    model_GNN = utils.update_params_total(model_GNN, sol_GNN)

    while (not done) and (t < cfg["environment"]["max_episode_steps"]):
        t += 1

        with torch.no_grad():
            actions = model_GNN(obs)
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
            final_state = new_obs[:,:cfg['environment']["n_connected_comp"]]
            ep_timesteps = t
            is_solved = int(infos['solved_task'])

        ep_rewards += reward

        obs = new_obs

    return [ep_rewards, ep_timesteps, is_solved]


def evaluate_batch(sol_GNN, model_GNN, cfg, envs, seeder=None):
    """Get rewards based on current es parameters"""

    seeds = seeder.next_batch(cfg["optimizer"]["pop_size"])
    ep_rewards = np.zeros((cfg["optimizer"]["pop_size"]))
    ep_timesteps = np.zeros((cfg["optimizer"]["pop_size"], 1))
    ep_solved = np.zeros((cfg["optimizer"]["pop_size"], 1))
    if cfg["utilities"]["multiprocessing"][0]:
        input_args = [(sol_GNN, ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(model_GNN), seeds[pop_id],
                       cfg, envs) for pop_id in range(cfg["optimizer"]["pop_size"])]
        with multiprocessing.Pool(processes=cfg["utilities"]["multiprocessing"][1]) as pool:
            pool_results = pool.starmap(simulate, input_args)
    else:
        pool_results = []
        for pop_id in range(cfg["optimizer"]["pop_size"]):
            pool_results.append(simulate(sol_GNN, ep_rewards[pop_id], ep_timesteps[pop_id],
                                         copy.deepcopy(model_GNN), seeds[pop_id], cfg, envs))
    for idx, result in enumerate(pool_results):
        ep_rewards[idx], ep_timesteps[idx], ep_solved[idx] = result

    return [ep_rewards, ep_timesteps, ep_solved]


class Trainer_GNN:
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

        print('Message-passing rounds: {}'.format(self.cfg["policy"]["mp_rounds"]))
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

        generated_graph = env.G

        # Set up models ================================================

        hyperparams = self.cfg["policy"]["hyperparameters"]

        model_type = import_module('models.' + self.cfg["policy"]["architecture"])
        create_model = getattr(model_type, 'create_model')

        W_1_dim = [self.node_feats_dim, hyperparams["initial_repr_out"][0]]

        phi_1 = create_model(input_size=2*hyperparams["initial_repr_out"][0],
                            output_size=hyperparams["initial_repr_out"][0],
                            layers=hyperparams["internal_dim"])

        mu = create_model(input_size=hyperparams["initial_repr_out"][0],
                          output_size=self.action_dim, layers=hyperparams["internal_dim"],
                          last_activation="softmax")
                          
        self.model_GNN = GNN_BL(self.cfg["policy"]["mp_rounds"], 
                                  W_1_dim, phi_1, mu=mu, seed=self.seed, cfg_env=self.cfg["environment"],
                                  graph=generated_graph)

        # Set up evolutions strategy ================================================

        init_params = None
        init_sigma = self.cfg["optimizer"]["sigma_init"]

        num_params = self.model_GNN.get_model_num_params()
        config["num_params_model"] = [num_params]
        self.es_GNN = CMAES(num_params, init_params, sigma_init=init_sigma, popsize=self.pop_size, seed=self.seed)

    def train(self):
        list_solved = [0]*20

        total_timesteps = 0

        for episode in range(self.cfg["simulation"]["num_episodes"]):

            sols_GNN = self.es_GNN.ask()      # get parameters from es
            ep_rewards = np.zeros((self.pop_size))
            ep_timesteps = np.zeros((self.pop_size, 1))
            ep_solved = np.zeros((self.pop_size, 1))
            seeds = self.seeder.next_batch(self.pop_size)

            if self.cfg["utilities"]["multiprocessing"][0]:
                input_args = [(sols_GNN[pop_id], ep_rewards[pop_id],
                               ep_timesteps[pop_id], copy.deepcopy(self.model_GNN), seeds[pop_id], self.cfg,
                               self.envs)
                              for pop_id in range(self.pop_size)]
                with multiprocessing.Pool(processes=self.cfg["utilities"]["multiprocessing"][1]) as pool:
                    pool_results = pool.starmap(simulate, input_args)
            else:
                pool_results = []
                for pop_id in range(self.pop_size):
                    input_args = (sols_GNN[pop_id],
                                  ep_rewards[pop_id], ep_timesteps[pop_id], copy.deepcopy(self.model_GNN),
                                  seeds[pop_id], self.cfg, self.envs)
                    pool_results.append(simulate(*input_args))

            for pop_id, result in enumerate(pool_results):
                ep_rewards[pop_id], ep_timesteps[pop_id], ep_solved[pop_id] = result
                total_timesteps += ep_timesteps[pop_id].sum()

            # Update parameters using es at the end of episode
            self.model_GNN, self.es_GNN = utils.run_es_update_params_total(self.model_GNN,
                                                                             self.es_GNN, ep_rewards)
            self.es_GNN = self.es_GNN[0]      # run_es_update_params_total() returns a list

            # Evaluate current params and update best reward
            if episode % self.cfg["simulation"]["eval_episodes_freq"] == 0:
                sols_GNN = np.array(self.es_GNN.current_param()).round(4)
                reward_eval, ep_timestep_eval, solved_eval = evaluate_batch(sols_GNN, copy.deepcopy(self.model_GNN),
                                                                            self.cfg, self.envs,
                                                                            seeder=self.seeder_eval)
                if self.cfg["simulation"]["exp_id"] != 0:
                    save_dict = {}
                    save_dict['params_sols'] = np.array(self.es_GNN.current_param()).round(4)
                    save_dict['sigma'] = np.array(self.es_GNN.current_sigma()).round(4)
                    save_dict['rms_sigma'] = np.array(self.es_GNN.rms_stdev()).round(4)
                    with open(os.path.join(self.exp_path, 'last_sols.pickle'), 'wb') as handle:
                        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if episode == 0:
                    best_reward_eval = np.mean(reward_eval)
                else:
                    if best_reward_eval < np.mean(reward_eval):
                        best_reward_eval = np.mean(reward_eval)
                        if self.cfg["simulation"]["exp_id"] != 0:
                            with open(os.path.join(self.exp_path, 'best_sols.pickle'), 'wb') as handle:
                                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print("|Episode: {}| Eval reward: {}| Percentage of solved: {}| Average timesteps: {}".format(episode, np.mean(reward_eval),
                                                                                                                np.mean(solved_eval),np.mean(ep_timestep_eval)))

                list_solved.pop(0)
                list_solved.append(np.mean(solved_eval))
                if np.mean(list_solved) > 0.95:
                    break
