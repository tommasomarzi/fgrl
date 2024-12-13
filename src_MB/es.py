import numpy as np


def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

class CMAES:
  '''CMA-ES wrapper.'''
  def __init__(self, num_params,      # number of model parameters
               init_params=None,
               sigma_init=0.10,       # initial standard deviation
               popsize=255,           # population size
               weight_decay=0.01,     # weight decay coefficient
               seed=0):               

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.popsize = popsize
    self.weight_decay = weight_decay
    self.solutions = None
    self.seed = seed
    
    import cma
    if init_params is None:
      init_values = list(np.random.uniform(-1., 1., self.num_params))
      self.es = cma.CMAEvolutionStrategy(init_values,
                                         self.sigma_init,
                                         {'popsize': self.popsize, 'seed': self.seed})
    else:
      self.es = cma.CMAEvolutionStrategy(init_params,
                                         self.sigma_init,
                                         {'popsize': self.popsize, 'seed': self.seed})

  def rms_stdev(self):
    sigma = self.es.result[6]
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''
    Returns a list of parameters.
    Note: es.ask() delivers new candidate solutions with shape [pop_size, num_params]
    '''
    self.solutions = np.array(self.es.ask())
    return self.solutions

  def tell(self, reward_table_result):
    '''
    es.tell(s, f(s)) updates the optimal instance by passing the respective function values.
    The objective function is the reward table with a L2 regularization of the solutions.
    '''
    reward_table = np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay
    self.es.tell(self.solutions, (-reward_table).tolist()) # convert minimizer to maximizer.

  def current_param(self):
    return self.es.result[5] # mean solution, presumably better with noise

  def current_sigma(self):
    return self.es.result[6] # current sigma

  def set_mu(self, mu):
    pass

  def best_param(self):
    return self.es.result[0] # best evaluated solution

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    r = self.es.result
    return (r[0], -r[1], -r[1], r[6])
