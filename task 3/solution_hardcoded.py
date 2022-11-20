import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process.kernels import Matern


import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from gpytorch.kernels import *
import gpytorch

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.pointsf = []
        self.pointsv = []
        self.pointsx = []


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.

        # return self.optimize_acquisition_function()



        # self.ucb = UpperConfidenceBound(self.gp, beta=0.1)
        # bounds = torch.stack([torch.tensor(domain[0][0][None], dtype=torch.double), torch.tensor(domain[0][1][None], dtype=torch.double)])
        # candidate, acq_value = optimize_acqf(
        #     self.ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20
        # )
        # # print(acq_value.detach().numpy())
        # print("candidate: {0:0.2f}, acq_value: {1:0.2f}".format(candidate.detach().numpy()[0][0], acq_value.detach().numpy()))
        # return candidate.detach().numpy()

        return 0 if len(self.pointsx) == 1 else self.pointsx[-1]+0.25


        # temp = self.pointsx[0] + np.random.randn() * 0.1
        # return 0 if temp < 0 else temp if temp < 5 else 5


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        pass
        # self.ucb = UpperConfidenceBound(self.gp, beta=0.1)
        # return 


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        x_ = np.array(x)
        if x_.ndim == 1:
            x = x_[0]
        elif x_.ndim == 2:
            x = x_[0][0]
        elif x_.ndim == 3:
            x = x_[0][0][0]

        f_ = np.array([f])
        if f_.ndim == 1:
            f = f_[0]
        elif f_.ndim == 2:
            f = f_[0][0]
        elif f_.ndim == 3:
            f = f_[0][0][0]

        v_ = np.array([v])
        if v_.ndim == 1:
            v = v_[0]
        elif v_.ndim == 2:
            v = v_[0][0]
        elif v_.ndim == 3:
            v = v_[0][0][0]


        print("x: {0:0.2f}, f: {1:0.2f}, v: {2:0.2f}".format(x, f, v))
        self.pointsx.append(x)
        self.pointsf.append(f)
        self.pointsv.append(v)

        # self.pointsx.append([x])
        # self.pointsf.append([f])
        # self.pointsv.append(v)
        # covar_module=gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=gpytorch.constraints.Interval(0.49, 0.51), outputscale_constraint=gpytorch.constraints.Interval(0.49, 0.51))
        # self.gp = SingleTaskGP(torch.tensor(self.pointsx, dtype=torch.double), torch.tensor(self.pointsf, dtype=torch.double), covar_module=covar_module)
        # mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        # fit_gpytorch_mll(mll)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # bounds = torch.stack([torch.tensor(domain[0][0][None], dtype=torch.double), torch.tensor(domain[0][1][None], dtype=torch.double)])
        # candidate, acq_value = optimize_acqf(
        #     self.ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20
        # )
        # return candidate.detach().numpy()

        best = -9
        bestx = 0
        finalv = -1
        for (x,f,v) in zip(self.pointsx, self.pointsf, self.pointsv):
            if f > best and v > 1.201:
                best = f
                bestx = x
                finalv = v

        print("best: {0:0.2f}, bestx: {1:0.2f}, finalv: {2:0.2f}".format(best, bestx, finalv))

        return np.array([bestx])
            



""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    n_dim = domain.shape[0]
    
    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
            1, n_dim)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)
    
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
