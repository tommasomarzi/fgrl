import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import xmltodict
from utilities import wrappers
import gym
from gym.envs.registration import register
from shutil import copyfile
from config import *
import json


def getMultiLevelGraph(name, n_hier_layers, exp_path=None):
    """
    Get hierarchical and agents graphs from xml file (original morphology) and json file (hard-coded clustering).
    """
    xml_path = os.path.join(XML_DIR, '{}.xml'.format(name))
    morphology = getGraphStructure(xml_path)
    directed_graph = torch.tensor(np.array([morphology[1:], np.arange(1, len(morphology))]), dtype=torch.long)
    a_graph_1 = torch_geometric.utils.to_undirected(directed_graph)

    a_graph = [a_graph_1]

    if exp_path is None:
        cluster_path = os.path.join(ENV_DIR, 'agents_hierarchy/' + str(n_hier_layers) + "_" + name.split('_')[0] + ".json")
    else:
                cluster_path = exp_path + "/" + str(n_hier_layers) + "_" + name.split('_')[0] + ".json"

    cluster_file = open(cluster_path)
    cluster_dict = json.load(cluster_file)
    cluster_file.close()

    for _, a_level in cluster_dict['agent_graph'].items():
        edge_tensor = torch.tensor(a_level, dtype=torch.int64)
        edge_tensor = torch_geometric.utils.to_undirected(edge_tensor)
        a_graph.append(edge_tensor)

    h_graph = []
    for _, h_level in cluster_dict['hier_graph'].items():
        edge_tensor = torch.tensor(h_level, dtype=torch.int64)
        h_graph.append(edge_tensor)

    assert(n_hier_layers == (len(a_graph)+1))
    assert(len(a_graph) == len(h_graph))

    return morphology, a_graph, h_graph


def run_es_update_params_total(es, rew):
    """
    Run the ES using given reward list.
    :param es: evolution strategy object
    :param rew: list of environment rewards
    :return: PyTorch model, list of ES objects
    """
    es.tell(rew)

    return [es]

def update_params_total(model, solution):
    """
    Update the model params for each layer using the given solution
    :param model: PyTorch model
    :param solution: list of ES solutions to be used as new Parameters
    :return: PyTorch model
    """

    model_params = solution
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


def run_es_update_params_from_layer(hier_levels, es, rew_list):
    """
    Run the ES using given reward list
    :param hier_levels: number of hierarchical levels
    :param es: list of evolution strategy objects
    :param rew_list: list of rewards
    :return: PyTorch model, list of ES objects
    """

    for lh in range(hier_levels):
        es[lh].tell(rew_list[lh])

    return es

def update_params_from_layer(model, hier_levels, solution):
    """
    Update the model params for each layer using the given solution
    :param model: PyTorch model
    :param hier_levels: number of hierarchical levels
    :param solution: list of ES solutions to be used as new Parameters
    :return: PyTorch model
    """

    for lh in range(hier_levels):
        model_params = solution[lh]
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


def makeEnvWrapper(env_name, obs_max_len=None, seed=0):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make("environments:%s-v0" % env_name)
        e.seed(seed)
        return wrappers.ModularEnvWrapper(e, obs_max_len)
    return helper


def findMaxChildren(env_names, graphs):
    """return the maximum number of children given a list of env names and their corresponding graph structures"""
    max_children = 0
    for name in env_names:
        most_frequent = max(graphs[name], key=graphs[name].count)
        max_children = max(max_children, graphs[name].count(most_frequent))
    return max_children


def registerEnvs(env_names, max_episode_steps, custom_xml):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if '.xml' in name:
                    paths_to_register.append(os.path.join(custom_xml, name))
    # register each env
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        env_file = env_name
        # create a copy of modular environment for custom xml model
        if not os.path.exists(os.path.join(ENV_DIR, '{}.py'.format(env_name))):
            # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
            copyfile(BASE_MODULAR_ENV_PATH, '{}.py'.format(os.path.join(ENV_DIR, env_name)))
        params = {'xml': os.path.abspath(xml)}
        # register with gym (check how it works)
        register(id=("%s-v0" % env_name),
                 max_episode_steps=max_episode_steps,
                 entry_point="environments.%s:ModularEnv" % env_file,
                 kwargs=params)
        env = wrappers.IdentityWrapper(gym.make("environments:%s-v0" % env_name))
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action
    return limb_obs_size, max_action


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q)-1)>1e-3):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
    theta = np.mod( theta + 2*np.pi, 2*np.pi )
    if theta > np.pi:
        theta =  2 * np.pi - theta
        r0    = -r0
    r = r0 * theta
    return r


def getGraphStructure(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""
    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch, self_idx)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml['mujoco']['worldbody']['body']
        assert not isinstance(root, list), 'worldbody can only contain one body (torso) for the current implementation, but found {}'.format(root)
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if 'walker' in os.path.basename(xml_file) and 'flipped' in os.path.basename(xml_file):
        parents[0] = -2
    return parents


def getGraphJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples (body_name, joint_name1, ...) for each body"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    def preorder(b):
        if 'joint' in b:
            if isinstance(b['joint'], list) and b['@name'] != 'torso':
                raise Exception("The given xml file does not follow the standard MuJoCo format.")
            elif not isinstance(b['joint'], list):
                b['joint'] = [b['joint']]
            joints.append([b['@name']])
            for j in b['joint']:
                joints[-1].append(j['@name'])
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml['mujoco']['worldbody']['body']
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml['mujoco']['actuator']['motor']
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m['@joint'])
    return joints
