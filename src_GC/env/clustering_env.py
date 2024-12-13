import gym
from gym import spaces
import numpy as np
import networkx as nx
from pygsp import graphs


class Graph_Clustering(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_params):
        self.n_colors = self.n_subgraphs = env_params['n_connected_comp']
        self.nodes_per_subgraph = env_params['nodes_per_cc']
        self.n_nodes = self.nodes_per_subgraph * self.n_subgraphs

        additional_obs = 3              # time step + (x,y)
        
        # generated connected community graph 
        starting_seed = 0
        while True:         # Ensure that the graph is connected
            self.G = graphs.Community(N = self.n_nodes, Nc = self.n_subgraphs,
                                        comm_sizes=[self.nodes_per_subgraph for _ in range(self.n_subgraphs)],
                                        seed=1 + starting_seed, min_deg=1)
            starting_seed += 1
            if (self.G.d == 0).sum() == 0:
                break
        self.xy = self.G.coords.astype(np.float32)

        # Actions: one for each color (left/noop/right)
        self.action_space = spaces.MultiDiscrete([3 for _ in range(self.n_nodes)])
        self.action_type = 'change'

        # Observations: one for each node (one-hot color + additional features)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_nodes, self.n_colors + additional_obs))

        # steps
        self.steps_threshold = env_params["max_episode_steps"]           # maximum number of steps
        self.steps = 0                                                   # current number of steps

        # states
        self.state = None

        # matrices for mincut reward        
        matrices = self.get_mincut_matrices()
        self.norm_adjacency_matrix = matrices[0]
        self.norm_degree_matrix = matrices[1]
        self.id_k = matrices[2]

    def step(self, actions):
        self.steps += 1

        # change state
        self.state = (self.state + actions.squeeze() - 1) % self.n_colors

        obs = self.generate_observation()

        additional_infos = {}

        done = False
        if self.steps == self.steps_threshold:
            done = True
            additional_infos['solved_task'] = False
        else:
            if self.check_solved_multiple_target():
                done = True
                additional_infos['solved_task'] = True
            
        if done:
            reward = self.get_mincut_reward()                  # mincut
            reward += (self.steps_threshold - self.steps)      # bonus
        else:
            reward = 0.

        return obs, reward, done, additional_infos

    def get_mincut_reward(self):
        cluster_matrix = np.zeros((self.n_nodes, self.n_subgraphs))
        cluster_matrix[np.arange(self.n_nodes), self.state] = 1
        l_c = - (np.trace(cluster_matrix.T @ self.norm_adjacency_matrix @ cluster_matrix) /
               np.trace(cluster_matrix.T @ self.norm_degree_matrix @ cluster_matrix))
        l_o = np.linalg.norm((cluster_matrix.T @ cluster_matrix)/np.linalg.norm(cluster_matrix.T @ cluster_matrix)
                             - self.id_k)
        return - (l_c + l_o)

    def check_solved_multiple_target(self):
        reshaped_state = self.state.reshape(self.n_subgraphs, self.nodes_per_subgraph)
        colors = []
        for subgraph in reshaped_state:
            vals, cnts = np.unique(subgraph, return_counts=True)
            if len(vals) > 1:
                return False
            colors.append(vals)
        return len(set(np.array(colors).squeeze())) == self.n_subgraphs

    def get_mincut_matrices(self):
        if self.G is None:
            G = nx.Graph()
            for i in range(self.n_subgraphs):
                H = nx.cycle_graph(self.nodes_per_subgraph)
                rel_H = {n: n + len(G.nodes) for n in H.nodes}
                H = nx.relabel_nodes(H, rel_H)
                G = nx.compose(G, H)
            adjacency_matrix = nx.adjacency_matrix(G).toarray()
            degrees = np.array([val for (node, val) in G.degree()])
        else:
            adjacency_matrix = self.G.W.toarray()
            degrees = self.G.d
        degree_matrix = np.eye(self.n_nodes) * np.array(degrees)
        D_12 = np.sqrt(np.linalg.inv(degree_matrix))
        norm_adjacency_matrix = D_12 @ adjacency_matrix @ D_12
        norm_degree_matrix = np.eye(self.n_nodes) * np.array(adjacency_matrix.sum(axis=1))
        id_k = np.eye(self.n_subgraphs)/np.sqrt(self.n_subgraphs)

        return norm_adjacency_matrix, norm_degree_matrix, id_k

    def generate_observation(self):
        obs = np.zeros((self.n_nodes, self.n_colors))
        obs[np.arange(self.n_nodes), self.state] = 1
        
        # add time steps 
        obs = np.concatenate((obs, np.zeros((self.n_nodes, 1))), axis=1)
        obs[np.arange(self.n_nodes), -1] = (1. - self.steps / self.steps_threshold)

        # add xy coords
        obs = np.concatenate((obs, self.xy), axis=1)
        return obs

    def reset(self):
        self.steps = 0
        self.state = np.random.randint(0, self.n_colors, self.n_nodes)
        while self.check_solved_multiple_target():
            self.state = np.random.randint(0, self.n_colors, self.n_nodes)
        obs = self.generate_observation()
        return obs

    def close(self):
        pass
