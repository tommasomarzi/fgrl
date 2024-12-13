import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import torch.nn as nn
import numpy as np
from numpy.linalg import norm


class FGRL(MessagePassing):
    """
    Class for the Feudal Graph Reinforcement Learning model. 
    """
    def __init__(self, lh, lr,  W_1_dim, rho, phi_1, psi, mu, seed=None,
                 goal_scale=False, cfg_env=None, graph=None):
        """
        All the lists are ordered in a bottom-up fashion (first element -> workers; last element -> top-level manager).
        Args:
            lh: number of hierarchical levels
            lr: number of message-passing rounds
            W_1_dim: list [in_dim, out_dim] for the weight matrix W_1.
            rho: list of trainable functions (initial representation).
            phi_1: list of lists of trainable functions (message functions for agent graphs).
            psi: list of trainable functions (goal functions).
            mu: trainable function (action generation).
            cfg_env: config file for the environment (required for generating the hierarchy).
            seed: seed for reproducibility.
            graph: generated graph structure of the environment.
        """
        super().__init__(aggr='mean', flow='source_to_target')
        if seed is not None:
            torch.manual_seed(seed)

        self.cfg_env = cfg_env
        self._Lh = lh
        self.hier_graph = self.generate_hier_graph()
        self.agent_graph = self.generate_agent_graph(graph)
        self._Lr = lr

        if goal_scale:
            self.fix_goals = True
            self.goal_scale = goal_scale
            self.goal_step_counter = 0
            self.current_goals = [None] * (self._Lh - 1)
        else:
            self.fix_goals = False

        self._W_1 = nn.Linear(W_1_dim[0], W_1_dim[1], bias=False)
        self._rho = nn.ModuleList(rho)
        self._phi_1 = nn.ModuleList([NeighbourFeats(phi_lh) for phi_lh in phi_1])
        self._psi = nn.ModuleList([GoalGen(f) for f in psi])
        self._mu = ActionGen(mu)

        self.current_h_hier = None          # current representations of the hierarchy


    def forward(self, f=None):
        """
        Args:
            f: additional node features with shape [num_workers, num_feats].
        """

        h_initial = self.current_h_hier
        h_final = h_initial.copy()

        for lr in range(self._Lr):
            for lh in range(self._Lh - 1):      # top-level manager has no neighbors
                h_final[lh] = self._phi_1[lh](h_final[lh], self.agent_graph[lh])

        goals = []
        for lh in reversed(range(self._Lh - 1)):
            if self.fix_goals and (self.goal_step_counter % self.goal_scale[lh] != 0):
                goals.append(self.current_goals[lh])
            else:
                if lh == (self._Lh - 2):        # top-level manager
                    goal_lh = self._psi[lh](h_final[lh+1], torch.cat((h_final[lh], h_initial[lh]), dim=1),
                                            self.hier_graph[lh])
                else:
                    goal_lh = self._psi[lh](goals[-1], torch.cat((h_final[lh], h_initial[lh]), dim=1),
                                            self.hier_graph[lh])
                goals.append(goal_lh)
                if self.fix_goals:
                    self.current_goals[lh] = goal_lh
        goals.reverse()

        self.goal_step_counter += 1

        actions = self._mu(torch.cat((goals[0], h_final[0]), dim=1)).numpy()

        return goals, actions

    def get_initial_repr(self, x):
        """
        Args:
            x: initial, raw representation of the workers concatenated to their features.
        """

        h_in_wrk = self._W_1(x.to(torch.float32))
        h_initial = [h_in_wrk]
        for lh in range(self._Lh - 1):
            h_transf = self._rho[lh](h_initial[-1])
            n_sup = len(self.hier_graph[lh][1].unique())
            h_lh = self.propagate(self.hier_graph[lh], x=h_transf, size=(h_transf.shape[0], n_sup))
            h_initial.append(h_lh)

        self.current_h_hier = h_initial         # save hierarchical representations for the next forward step

        return h_initial

    def get_reward(self, new_obs, feats=None, par_goals=None):
        """
        Args:
            new_obs: new observations of the limbs.
            feats: nodes feats.
            par_goals: [goals, new_obs].
        """

        if feats is not None:
            x = torch.cat([new_obs, feats], dim=1)
        else:
            x = new_obs

        h_initial = self.get_initial_repr(x.to(torch.float32))

        goals, new_obs = par_goals
        rewards = self.reward_with_goals(h_initial, goals, new_obs)

        return rewards

    def reward_with_goals(self, h_initial, goals, new_obs):
        rewards = []

        new_obs = new_obs[:, :len(goals[1])]
        one_hot_goals = np.zeros_like(goals[0].numpy())
        one_hot_goals[np.arange(len(goals[0])), np.argmax(goals[0].numpy(), axis=1)] = 1
        rewards_wrk = self.compute_similarity(one_hot_goals, new_obs)
        rewards.append(rewards_wrk)

        subgraph_mean = new_obs.reshape(self.cfg_env["n_connected_comp"],
                                        self.cfg_env["nodes_per_cc"],
                                        self.cfg_env["n_connected_comp"]).mean(axis=1)
        one_hot_goals_sub = np.zeros_like(goals[1].numpy())
        one_hot_goals_sub[np.arange(len(goals[1])), np.argmax(goals[1].numpy(), axis=1)] = 1

        rewards_sub = self.compute_similarity(one_hot_goals_sub, subgraph_mean)
        rewards.append(rewards_sub)

        return rewards

    def compute_similarity(self, vec_1, vec_2, shift = -0.5):
        res = np.sum(np.multiply(vec_1, vec_2), axis=1) / np.multiply(norm(vec_1, axis=1), norm(vec_2, axis=1))
        res = np.mean(res) + shift
        return res

    def generate_agent_graph(self, graph):
        agent_graph = []

        # worker graph
        from networkx import from_numpy_matrix
        worker_graph = from_networkx(from_numpy_matrix(graph.W.toarray()))
        agent_graph.append(worker_graph.edge_index)

        # submanager graph (complete graph)
        from networkx import complete_graph
        submanager_graph = complete_graph(self.cfg_env["n_connected_comp"])
        submanager_graph = from_networkx(submanager_graph).edge_index
        agent_graph.append(submanager_graph)

        # manager graph
        agent_graph.append(None)

        return agent_graph

    def generate_hier_graph(self):
        hier_graph = []

        workers = np.arange(self.cfg_env["nodes_per_cc"]*self.cfg_env["n_connected_comp"])
        sub_managers = np.array([np.repeat(n, self.cfg_env["nodes_per_cc"])
                                 for n in range(self.cfg_env["n_connected_comp"])]).flatten()
        first_level = np.concatenate(([workers], [sub_managers]), axis=0)
        hier_graph.append(torch.tensor(first_level, dtype=torch.int64))

        sub_managers = np.arange(self.cfg_env["n_connected_comp"])
        managers = np.repeat(0, self.cfg_env["n_connected_comp"])
        second_level = np.concatenate(([sub_managers], [managers]), axis=0)
        hier_graph.append(torch.tensor(second_level, dtype=torch.int64))

        return hier_graph

    def get_model_num_params(self):
        """
        Return sum of the number of parameters of trainable functions of all the FGRL model.
        """
        num_pars = 0

        num_pars += sum([i.numel() for i in self._W_1.state_dict().values()])
        num_pars += sum([i.numel() for i in self._mu.state_dict().values()])

        for lh in range(self._Lh - 1):
            num_pars += sum([i.numel() for i in self._rho[lh].state_dict().values()])
            num_pars += sum([i.numel() for i in self._psi[lh].state_dict().values()])

        for lh in range(self._Lh): 
                if lh < (self._Lh - 1):     # top-level manager has no neighbors
                    num_pars += sum([i.numel() for i in self._phi_1[lh].state_dict().values()])

        return num_pars

    def get_model_funcs(self):
        """
        Return list of trainable functions of all the FGRL model.
        """
        functions = []

        functions.append(self._W_1)
        functions.append(self._mu)

        for lh in range(self._Lh - 1):
            functions.append(self._rho[lh - 1])
            functions.append(self._psi[lh - 1])

        for lh in range(self._Lh):
            if lh < (self._Lh - 1):
                functions.append(self._phi_1[lh])

        return nn.ModuleList(functions)

    def get_num_params_from_layer(self, lh):
        """
        Return sum of the number of parameters of trainable functions given a layer lh of the FGRL model.
        """
        num_pars = 0

        if lh == 0:
            num_pars += sum([i.numel() for i in self._W_1.state_dict().values()])
            num_pars += sum([i.numel() for i in self._mu.state_dict().values()])
        else:
            num_pars += sum([i.numel() for i in self._rho[lh - 1].state_dict().values()])
            num_pars += sum([i.numel() for i in self._psi[lh - 1].state_dict().values()])

        if lh < (self._Lh - 1):
            num_pars += sum([i.numel() for i in self._phi_1[lh].state_dict().values()])

        return num_pars

    def get_funcs_from_layer(self, lh):
        """
        Return list of trainable functions given a layer lh of the FGRL model.
        """
        functions = []

        if lh == 0:
            functions.append(self._W_1)
            functions.append(self._mu)
        else:
            functions.append(self._rho[lh - 1])
            functions.append(self._psi[lh - 1])

        if lh < (self._Lh - 1):
            functions.append(self._phi_1[lh])

        return nn.ModuleList(functions)


class GoalGen(MessagePassing):
    def __init__(self, net):
        super().__init__(aggr='add')
        self.message_network = net

    def forward(self, x_sup, x_sub, edge_index):
        messages = self.propagate(edge_index.flip(1,0), x=(x_sup, x_sub), size=(x_sup.size(0), x_sub.size(0)))
        return messages

    def message(self, x_i, x_j):
        out = self.message_network(torch.cat([x_j, x_i], dim=1))
        return out


class NeighbourFeats(MessagePassing):
    def __init__(self, net):
        super().__init__(aggr='add')
        self.message_network = net

    def forward(self, x, edge_index):
        messages = self.propagate(edge_index, x=x)
        return messages

    def message(self, x_i, x_j):
        out = self.message_network(torch.cat([x_i, x_j], dim=1))
        return out


class ActionGen(nn.Module):
    def __init__(self, act_net, msg_layers=None):
        super().__init__()
        self.act_network = act_net
        self.msg_layers = msg_layers

    def forward(self, goals, node_idx=None):
        """
        :param goals: goals generated by manager
        :return: actions
        """

        if node_idx is None:
            return self.act_network(goals)

        return self.act_network(goals[node_idx])
