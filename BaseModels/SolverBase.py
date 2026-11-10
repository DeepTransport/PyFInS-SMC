import abc

class SolverBase(abc.ABC):
    """
    Abstract solver interface.
    """

    def __init__(self):
        self.block_mode = False # store as-is; subclass adapts/validates
        self.n_parameter = 0
        self.n_data = 0
        self.n_qoi = 0
        self.sensor2data = None

    @abc.abstractmethod
    def solve(self, x):
        raise NotImplementedError

class SolverAdjoint(SolverBase):
    """
    Abstract likelihood interface.
    """
    
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def linearized_observation(self, delta_params_phys):
        raise NotImplementedError
        
    @abc.abstractmethod
    def linearized_observation_transpose(self, delta_y):
        raise NotImplementedError

    @abc.abstractmethod
    def linearized_qoi_transpose(self):
        raise NotImplementedError


class SolverAdjointCache(SolverAdjoint):
    """
    Abstract likelihood interface.
    """
    
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def solve_and_cache(self, params_phys):
        raise NotImplementedError
    
    