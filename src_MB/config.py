ENV_DIR = './environments'
XML_DIR = './environments/xmls'
BASE_MODULAR_ENV_PATH = './environments/ModularEnv.py'
DATA_DIR = './results'


class Config:
    def __init__(self):
        self.exp_id = 1                # identifier for the experiment
        self.seed = self.exp_id
        self.morphologies = ['hopper_5']
        self.num_episodes = 100
        self.eval_episodes = 1
        self.max_episode_steps = 1000
        self.max_timesteps = 20e6
        self.architecture = 'fc'

        self.sigma_init = 0.25
        self.pop_size = 64                # population size (lamda) CMA-ES

        self.use_baseline = [False, "MLP"]      # baseline (True/False) and baseline type (FDS/GNN/DS/MLP)

        self.hier_levels = 2
        self.mp_rounds = 2
        self.enable_features = True

        self.single_optimizer = False        # single ES (ablation)
        self.no_intrinsic_reward = False    # no intrinsic rewards (ablation)

        self.custom_xml = None

        self.multiprocessing = [False, 1]            # second element is the number of CPUs

        # # FC Network hyperparameters
        internal_dim = [30]
        self.dim_representation = 20
        self.rho_layers = internal_dim
        self.phi_1_layers = internal_dim
        self.psi_layers = internal_dim
        self.mu_layers = internal_dim
        self.mu_out = 1
