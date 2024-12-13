import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import numpy as np
from numpy.linalg import norm


class Feudal_BL(MessagePassing):
    """
    Class for the feudal baseline, i.e. use only the hierarchical setup.
    """
    def __init__(self, lh, W_1_dim, rho, psi, mu, seed=None):
        """
        All the lists are ordered in a bottom-up fashion (first element -> workers; last element -> top-level manager).
        Args:
            lh: number of hierarchical levels
            W_1_dim: list [in_dim, out_dim] for the weight matrix W_1.
            rho: list of trainable functions (initial representation).
            psi: list of trainable functions (goal functions).
            mu: trainable function (action generation).
        """
        super().__init__(aggr='add', flow='source_to_target')
        if seed is not None:
            torch.manual_seed(seed)

        self._Lh = lh

        self._W_1 = nn.Linear(W_1_dim[0], W_1_dim[1], bias=False)
        self._rho = nn.ModuleList(rho)
        self._psi = nn.ModuleList([GoalGen(f) for f in psi])
        self._mu = ActionGen(mu)

        self.current_h_hier = None

    def forward(self, hier_graph):
        """
        Remark: the current state is encoded in self.current_h_hier
        Args:
            hier_graph: list of torch tensors representing the hierarchical graph (each tensor represents the graph
                        lh -> lh+1).
        """

        h_final = self.current_h_hier

        goals = []
        for lh in reversed(range(self._Lh - 1)):
            if lh == (self._Lh - 2):
                goal_lh = self._psi[lh](h_final[lh+1], h_final[lh], hier_graph[lh])
            else:
                goal_lh = self._psi[lh](goals[-1], h_final[lh], hier_graph[lh])
            goals.append(goal_lh)
        goals.reverse()

        actions = self._mu(torch.cat((goals[0], h_final[0]), dim=1)).numpy()

        return goals, actions

    def get_initial_repr(self, x, hier_graph):
        """
        Args:
            x: initial, raw representation of the workers concatenated to their features.
            hier_graph: list of torch tensors representing the hierarchical graph (each tensor represents the graph
                        lh -> lh+1).
        """

        h_in_wrk = self._W_1(x.to(torch.float32))
        h_initial = [h_in_wrk]
        for lh in range(self._Lh - 1):
            h_transf = self._rho[lh](h_initial[-1])
            n_sup = len(hier_graph[lh][1].unique())
            h_lh = self.propagate(hier_graph[lh], x=h_transf, size=(h_transf.shape[0], n_sup))
            h_initial.append(h_lh)

        self.current_h_hier = h_initial

        return h_initial

    def get_reward(self, goals, obs_limbs, new_obs, hier_graph):
        """
        Args:
            goals: list of numpy arrays containing the goals (each element of the list is associated to a level).
            obs_limbs: difference in the observations of the limbs (new_state - old_state).
            new_obs: new observations of the limbs.
            hier_graph: list of torch tensors representing the hierarchical graph (each tensor represents the graph
                        lh -> lh+1).
        """

        h_initial = self.get_initial_repr(new_obs.to(torch.float32), hier_graph)

        rewards = []

        reward_wrk = []
        for goal_wrk, obs_limb in zip(goals[0], obs_limbs):
            if np.isclose(goal_wrk.numpy().sum(), 0) | np.isclose(obs_limb.sum(), 0):
                reward_wrk.append(0)
            else:
                reward_wrk.append(1 + (np.dot(goal_wrk, obs_limb) / (norm(goal_wrk) * norm(obs_limb))))
        rewards.append(np.mean(np.array(reward_wrk)))

        for lh in range(self._Lh)[1:-1]:
            reward_sub_lh = []
            for goal_sub, obs_sub in zip(goals[lh], h_initial[lh]):
                if np.isclose(goal_sub.numpy().sum(), 0) | np.isclose((obs_sub.detach().numpy()).sum(), 0):
                    reward_sub_lh.append(0)
                else:
                    reward_sub_lh.append(1 + (np.dot(goal_sub, obs_sub.detach().numpy())
                                         / (norm(goal_sub) * norm(obs_sub.detach().numpy()))))
            rewards.append(np.mean(reward_sub_lh))

        return rewards

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

        return nn.ModuleList(functions)


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