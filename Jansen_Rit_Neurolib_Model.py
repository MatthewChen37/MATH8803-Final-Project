# Modified from the tutorial: https://neurolib-dev.github.io/examples/example-4.1-multimodel-custom-model/#jansen-rit-model

import matplotlib.pyplot as plt
import numpy as np
import symengine as se
from IPython.display import display
from jitcdde import input as system_input
from neurolib.models.multimodel import MultiModel, ThalamicNode
from neurolib.models.multimodel.builder.base.constants import LAMBDA_SPEED
from neurolib.models.multimodel.builder.base.network import Network, Node
from neurolib.models.multimodel.builder.base.neural_mass import NeuralMass
from neurolib.utils.functions import getPowerSpectrum
from neurolib.utils.stimulus import Input, OrnsteinUhlenbeckProcess, StepInput

class UniformlyDistributedNoise(Input):
    """
    Uniformly distributed noise process between two values.
    """

    def __init__(self, low, high, n=1, seed=None):
        # save arguments as attributes for later
        self.low = low
        self.high = high
        # init super
        super().__init__(n=n, seed=seed)

    def generate_input(self, duration, dt):
        # generate time vector
        self._get_times(duration=duration, dt=dt)
        # generate noise process itself with the correct shape
        # as (time steps x num. processes)
        return np.random.uniform(self.low, self.high, (self.n, self.times.shape[0]))
    
# let us build a proper hierarchy, i.e. we firstly build a Jansen-Rit mass
class SingleJansenRitMass(NeuralMass):
    """
    Single Jansen-Rit mass implementing whole three population dynamics.

    Reference:
        Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked potential
        generation in a mathematical model of coupled cortical columns. Biological cybernetics,
        73(4), 357-366.
    """

    # all these attributes are compulsory to fill in
    name = "Jansen-Rit mass"
    label = "JRmass"

    num_state_variables = 7  # 6 ODEs + firing rate coupling variable
    num_noise_variables = 1  # single external input
    # NOTE 
    # external inputs (so-called noise_variables) are typically background noise drive in models,
    # however, this can be any type of stimulus - periodic stimulus, step stimulus, square pulse,
    # anything. Therefore you may want to add more stimuli, e.g. for Jansen-Rit model three to each
    # of its population. Here we do not stimulate our Jansen-Rit model, so only use actual noise
    # drive to excitatory interneuron population.
    # as dictionary {index of state var: it's name}
    coupling_variables = {6: "r_mean_EXC"}
    # as list
    state_variable_names = [
        "v_pyr",
        "dv_pyr",
        "v_exc",
        "dv_exc",
        "v_inh",
        "dv_inh",
        # to comply with other `MultiModel` nodes
        "r_mean_EXC",
    ]
    # as list
    # note on parameters C1 - C4 - all papers use one C and C1-C4 are
    # defined as various rations of C, typically: C1 = C, C2 = 0.8*C
    # C3 = C4 = 0.25*C, therefore we use only `C` and scale it in the
    # dynamics definition
    required_params = [
        "A",
        "a",
        "B",
        "b",
        "C",
        "v_max",
        "v0",
        "r",
        "lambda",
    ]
    # list of required couplings when part of a `Node` or `Network`
    # `network_exc_exc` is the default excitatory coupling between nodes
    required_couplings = ["network_exc_exc"]
    # here we define the default noise input to Jansen-Rit model (this can be changed later)
    # for a quick test, we follow the original Jansen and Rit paper and use uniformly distributed
    # noise between 120 - 320 Hz; but we do it in kHz, hence 0.12 - 0.32
    # fix seed for reproducibility
    _noise_input = [UniformlyDistributedNoise(low=0.12, high=0.32, seed=42)]

    def _sigmoid(self, x):
        """
        Sigmoidal transfer function which is the same for all populations.
        """
        # notes:
        # - all parameters are accessible as self.params - it is a dictionary
        # - mathematical definition (ODEs) is done in symbolic mathematics - all functions have to be
        #   imported from `symengine` module, hence se.exp which is a symbolic exponential function
        return self.params["v_max"] / (
            1.0 + se.exp(self.params["r"] * (self.params["v0"] - x))
        )

    def __init__(self, params=None, seed=None):
        # init this `NeuralMass` - use passed parameters or default ones
        # parameters are now accessible as self.params, seed as self.seed
        super().__init__(params=params or JR_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        # random in average potentials around zero
        self.initial_state = (
            np.random.normal(size=self.num_state_variables)
            # * np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0])
        ).tolist()

    def _derivatives(self, coupling_variables):
        """
        Here the magic happens: dynamics is defined here using symbolic maths package symengine.
        """
        # first, we need to unwrap state vector
        (
            v_pyr,
            dv_pyr,
            v_exc,
            dv_exc,
            v_inh,
            dv_inh,
            firing_rate,
        ) = self._unwrap_state_vector()  # this function does everything for us
        # now we need to write down our dynamics
        # PYR dynamics
        d_v_pyr = dv_pyr
        d_dv_pyr = (
            self.params["A"] * self.params["a"] * self._sigmoid(v_exc - v_inh)
            - 2 * self.params["a"] * dv_pyr
            - self.params["a"] ** 2 * v_pyr
        )
        # EXC dynamics: system input comes into play here
        d_v_exc = dv_exc
        d_dv_exc = (
            self.params["A"]
            * self.params["a"]
            * (
                # system input as function from jitcdde (also in symengine) with proper index:
                # in our case we have only one noise input (can be more), so index 0
                system_input(self.noise_input_idx[0])
                # C2 = 0.8*C, C1 = C
                + (0.8 * self.params["C"]) * self._sigmoid(self.params["C"] * v_pyr)
            )
            - 2 * self.params["a"] * dv_exc
            - self.params["a"] ** 2 * v_exc
        )
        # INH dynamics
        d_v_inh = dv_inh
        d_dv_inh = (
            self.params["B"] * self.params["b"]
            # C3 = C4 = 0.25 * C
            * (0.25 * self.params["C"])
            * self._sigmoid((0.25 * self.params["C"]) * v_pyr)
            - 2 * self.params["b"] * dv_inh
            - self.params["b"] ** 2 * v_inh
        )
        # firing rate computation
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed point dynamics
        firing_rate_now = self._sigmoid(v_exc - v_inh)
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        # now just return a list of derivatives in the correct order
        return [d_v_pyr, d_dv_pyr, d_v_exc, d_dv_exc, d_v_inh, d_dv_inh, d_firing_rate]
    

class JansenRitNode(Node):
    """
    Jansen-Rit node with 1 neural mass representing 3 population model.
    """

    name = "Jansen-Rit node"
    label = "JRnode"

    # if Node is integrated isolated, what network input we should use
    # zero by default = no network input for one-node model
    default_network_coupling = {"network_exc_exc": 0.0}

    # default output is the firing rate of pyramidal population
    default_output = "r_mean_EXC"

    # list of all variables that are accessible as outputs
    output_vars = ["r_mean_EXC", "v_pyr", "v_exc", "v_inh"]

    def __init__(self, params=None, seed=None):
        # in `Node` __init__, the list of masses is created and passed
        jr_mass = SingleJansenRitMass(params=params, seed=seed)
        # each mass has to have index, in this case it is simply 0
        jr_mass.index = 0
        # call super and properly initialize a Node
        super().__init__(neural_masses=[jr_mass])

        self.excitatory_masses = np.array([0])

    def _sync(self):
        # this function typically defines the coupling between masses
        # within one node, but in our case there is nothing to define
        return []