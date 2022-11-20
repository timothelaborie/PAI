from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import Branin
import os
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
import numpy as np
from botorch.test_functions import Hartmann
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from matplotlib import pyplot as plt
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import *
from gpytorch.priors import *
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from botorch.models import FixedNoiseGP, ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.stats import norm

import time
import warnings

warnings.filterwarnings("ignore")

dtype = torch.double
torch.set_default_dtype(torch.float64)

kernel_f = MaternKernel(nu = 2.5, lengthscale_prior = NormalPrior(0.5, scale = 0.01)) #+ WhiteNoiseKernel()
kernel_v = MaternKernel(nu = 2.5, lengthscale_prior = NormalPrior(0.5, scale = 0.01))
class GPF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(constant_prior = NormalPrior(0.0, np.sqrt(0.5)))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_f)#.add_diagonal(0.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPV(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPV, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(constant_prior = NormalPrior(1.5, np.sqrt(2)))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_v)#.add_diagonal(0.0001)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



domain = np.array([[0, 5]])

""" Solution """
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        self.xs = []
        self.fs = []
        self.vs = []
        self.wf = []
        self.v_var = np.sqrt(2)
        self.f_var = 0.5
        self.vp = 0

        self.violations = 0
        self.rec = 'normal'

        self.mll_ei = None
        self.model_ei = None
        self.trial_index = 0

        # define a feasibility-weighted objective for optimization
        self.constrained_obj = ConstrainedMCObjective(
            objective=self.obj_callable,
            constraints=[self.constraint_callable],
        )

        self.gs = GenerationStrategy(
            steps=[
                # Quasi-random initialization step
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=5,  # How many trials should be produced from this generation step
                ),
            ]
        )
        self.ax_client = AxClient(generation_strategy=self.gs,verbose_logging=False)
        # Setup the experiment
        self.ax_client.create_experiment(
            name="task3",
            overwrite_existing_experiment=True,
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
                    # Otherwise, the parameter would be inferred as an integer range.
                    "bounds": [0.0, 5.0],
                },
            ],
            objectives={
                "f": ObjectiveProperties(minimize=False),
                "v": ObjectiveProperties(minimize=False),
            },
        )

    def obj_callable(self, Z):
        return Z[..., 0]


    def constraint_callable(self, Z):
        return Z[..., 1]


    def initialize_model(self, train_x, train_obj, train_con, state_dict=None):
        noise_f = torch.full_like(train_obj, 0.15)
        noise_v = torch.full_like(train_con, 0.0001)
        # define models for objective and constraint
        # print(f"train_x: {train_x}, train_obj: {train_obj}, train_yvar: {train_con}")
        # model_obj = SingleTaskGP(train_x, train_obj, covar_module=kernel_f).to(train_x)
        # model_con = SingleTaskGP(train_x, train_con, covar_module = kernel_v, mean_module=ConstantMean(constant_prior = NormalPrior(1.5, 0.0001))).to(train_x)
        model_obj = FixedNoiseGP(train_x, train_obj, train_Yvar = noise_f, covar_module=kernel_f).to(train_x)
        model_con = FixedNoiseGP(train_x, train_con, train_Yvar = noise_v, covar_module = kernel_v, mean_module=ConstantMean(constant_prior = NormalPrior(1.5, np.sqrt(self.v_var)))).to(train_x)

        # combine into a multi-output GP model
        model = ModelListGP(model_obj, model_con)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        return mll, model

    # def get_v_pred(self, x):
    #     # print(f" getting v preds")
    #     self.gpv.eval()
    #     self.likelihoodv.eval()

    #     # Test points are regularly spaced along [0,1]
    #     # Make predictions by feeding model through likelihood
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         f_preds = self.likelihoodv(self.gpv(x), noise = torch.ones(x.shape[0])*0.0001)
    #     #print(f"f_preds incoming: -------------- \n \n \n \n{f_preds}")
    #     p = torch.distributions.normal.Normal(loc = f_preds.mean, scale=f_preds.stddev).cdf(torch.ones_like(f_preds.mean)*1.2)
    #     # gp_mean = f_preds.mean.cpu().numpy()
    #     # gp_std = f_preds.stddev.detach().cpu().numpy()
    #     return 1.0-p

    # def get_f_pred(self, x):
    #     # print(f" getting v preds")
    #     self.gpf.eval()
    #     self.likelihoodf.eval()
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         f_preds = self.likelihoodf(self.gpf(x), noise = torch.ones(x.shape[0])*0.15)
    #     return f_preds.mean

    def get_v_pred(self, x):

        try:
            x = x.numpy()
            x = x.reshape(-1, 1)
            mu, s = self.gpv.predict(x, return_std=True)
            p = 1 - norm.cdf(1.2, mu, s)
            return p
        except:
            x = x.reshape(-1, 1)
            mu, s = self.gpv.predict(x, return_std=True)
            p = 1 - norm.cdf(1.2, mu, s)
            return p

    def get_f_pred(self, x):
        try:
            x = x.numpy()
            x = x.reshape(-1, 1)
            return self.gpf.predict(x)
        except:
            x = x.reshape(-1, 1)
            return self.gpf.predict(x)


    def opt_get(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        # print(f"getting new candidates")
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.Tensor([0.0, 5.0]).reshape(2, 1),
            q=16,
            num_restarts=5,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values 
        new_x = candidates.detach()
        # print(f"got new candidates")
        return new_x



    def outcome_constraint(self, x):
        """L1 constraint; feasible if less than or equal to zero."""

        return self.get_v_pred(x) > 0.9

    def weighted_obj(self, f, x, v):
        """Feasibility weighted objective; zero if not feasible."""
        p = self.get_v_pred(x)
        # return f*p*(v - 1.2), p
        # return f*(v - 1.2)*p, p
        # return f*(v > 1.2)*p, p
        if f < 0:
            return f, p
        else:
            return f*p*(v-1.2), p
        # return f * (self.outcome_constraint(x) <= 0.0).type_as(f)
        # return f * self.get_v_pred(x)
        # return f*self.outcome_constraint(x)
        #return f*p, p

        
    def next_recommendation(self):
        # print("Get recommendation----------------------------------")
        train_x_ei, train_obj_ei, train_con_ei = torch.Tensor(self.xs).unsqueeze(1), torch.Tensor(self.fs).unsqueeze(1), torch.Tensor(self.vs).unsqueeze(1)
        standardize(train_obj_ei)
        standardize(train_con_ei)
        self.mll_ei, self.model_ei = self.initialize_model(train_x_ei, train_obj_ei, train_con_ei)
        fit_gpytorch_model(self.mll_ei)
        constraints = {1: (1.2, None)}
        best_f = np.max(self.fs)
        cei = ConstrainedExpectedImprovement(self.model_ei, 0.2, 0, constraints)
        candidate, acq_value = optimize_acqf(
                                acq_function=cei, 
                                bounds=torch.Tensor([0.0, 5.0]).reshape(2, 1),
                                 q=1,
                                  num_restarts=5,
                                raw_samples=20,
                            )
        p = self.get_v_pred(candidate)
        if self.violations:
            thresh = 0.75
        else:
            thresh = 0.5
        if len(self.xs) == 20 and not self.violations:
            thresh = -1
        if p > thresh:
            x = candidate
            self.rec = 'normal'
        else:
            self.rec = 'con'
            qmc_sampler = SobolQMCNormalSampler(num_samples=512)
            noise_v = torch.full_like(train_con_ei, 0.15)
            self.model_v = FixedNoiseGP(train_x_ei, train_con_ei, train_Yvar = noise_v, covar_module = kernel_v, mean_module=ConstantMean(constant_prior = NormalPrior(1.5, np.sqrt(np.sqrt(self.v_var))))).to(train_x_ei)
            self.mll_v = ExactMarginalLogLikelihood(self.model_v.likelihood, self.model_v)
            fit_gpytorch_model(self.mll_v)
            # qEIv = qExpectedImprovement(
            #                       model=self.model_v, 
            #                       best_f=np.max(self.vs),
            #                       sampler=qmc_sampler, 
            #                       )

            qEIv = qNoisyExpectedImprovement(
                                        model=self.model_v, 
                                        X_baseline=train_x_ei,
                                        sampler=qmc_sampler, 

                                    )
            x, acq_value = optimize_acqf(
                                        qEIv, 
                                        bounds=torch.Tensor([0.0, 5.0]).reshape(2, 1),
                                        q=64, num_restarts=5,
                                         raw_samples=512,
                                    )
            v_preds = self.get_v_pred(x)
            if v_preds.max() > thresh:
                x = x[v_preds > thresh]
                v_preds = v_preds[v_preds > thresh]
                f_preds = self.get_f_pred(x)
                self.vp = v_preds.max()
                w_obj = f_preds*v_preds
                x = x[np.argmax(w_obj)]
            else:
                self.vp = 0
                x = np.linspace(0.0, 5.0, 100)
                # print(f"\n got this far ................. x {x}")
                p = self.get_v_pred(x)
                # print(f"linspace v --------- \n \n {p}")
                x = x[p > thresh]
                f = self.get_f_pred(x)
                x = x[np.argmax(f)]
            
        

        return np.atleast_2d(x)


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
        # print(f"f: {f}, v: {v} x: {x.flatten()[0]}")
        # print("--------------add data point-------------------------------------------")
        self.xs.append(x.flatten()[0])
        self.vs.append(v.flatten()[0])
        self.fs.append(f.flatten()[0])
        
        if v.flatten()[0] < 1.2:
            self.violations = 1
            print(f"xxxxxxxxxxxx-------------xxxxxxxxxxxxxxxxx \n")
            print(f"violation from {self.rec}, p was {self.vp} \n")

        xtrain = np.array(self.xs).reshape(-1, 1)
        vtrain = np.array(self.vs)
        ftrain = np.array(self.fs)

        kernel = Matern(length_scale=0.5, nu=2.5, length_scale_bounds="fixed") + ConstantKernel(1.5, constant_value_bounds="fixed") + WhiteKernel(noise_level = 0.0001, noise_level_bounds="fixed") + ConstantKernel(np.sqrt(2))
        self.gpv = GaussianProcessRegressor(kernel=kernel)
        self.gpv.fit(xtrain, vtrain)


        kernel = Matern(length_scale=0.5, nu=2.5, length_scale_bounds="fixed") +  WhiteKernel(noise_level = 0.15, noise_level_bounds="fixed") #+ ConstantKernel(0.5)
        self.gpf = GaussianProcessRegressor(kernel=kernel)
        self.gpf.fit(xtrain, ftrain)

        # train the gpv model
        # xtrain = torch.tensor(self.xs)
        # ytrain_v = torch.tensor(self.vs)
        # ytrain_f = torch.tensor(self.fs)
        # # standardize(ytrain)
        # noises_f = torch.ones(len(self.xs))*0.15
        # noises_v = torch.ones(len(self.xs))*0.0001
        # self.likelihoodv = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noises_v)
        # self.gpv = GPV(xtrain, ytrain_v, self.likelihoodv)

        # self.likelihoodf = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noises_f)
        # self.gpf = GPF(xtrain, ytrain_f, self.likelihoodf)

        # # Find optimal model hyperparameters
        # self.gpv.train()
        # self.likelihoodv.train()

        # self.gpf.train()
        # self.likelihoodf.train()

        # # Use the adam optimizer
        # optimizer_v = torch.optim.Adam(self.gpv.parameters(), lr=0.05)
        # optimizer_f = torch.optim.Adam(self.gpf.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

        # # "Loss" for GPs - the marginal log likelihood
        # self.mll_v = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoodv, self.gpv)
        # self.mll_f = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoodf, self.gpf)

        # for i in range(len(self.xs) + 1):
        #     # Zero gradients from previous iteration
        #     optimizer_v.zero_grad()
        #     optimizer_f.zero_grad()
        #     # Output from model
        #     output_v = self.gpv(xtrain)
        #     output_f = self.gpf(xtrain)
        #     # Calc loss and backprop gradients
        #     loss_v = -self.mll_v(output_v, ytrain_v)
        #     loss_f = -self.mll_f(output_f, ytrain_f)
        #     loss_v.backward()
        #     loss_f.backward()
        #     # print('Iter %d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     #     i + 1, loss.item(),
        #     #     self.gpv.covar_module.base_kernel.lengthscale.item(),
        #     #     self.gpv.likelihood.noise.item()
        #     # ))
        #     optimizer_v.step()
        #     optimizer_f.step()



        # wf, p = self.weighted_obj(torch.Tensor(f), torch.Tensor(x), torch.Tensor(v))
        # print(f"f is {f}, wf is {wf} p is {p}\n \n")
        # # print(f"ws: --------------------\n \n\ \n{wf}")
        # if v.flatten()[0] < 1.2:
        #     print(f"unsafe: {v.flatten()[0]} p: {p}, f: {f.flatten()[0]}")
        #     wf = 0.0
        # self.wf.append(wf)
        # if len(self.xs) > 1 and len(self.xs) < 3:

        #     p = p.numpy()[0]
        #     f = f.flatten()[0]
        #     # print(f"here: {f*p} -------------------------")
        #     pf = p*f

            
        #     self.ax_client.complete_trial(trial_index=self.trial_index, raw_data={"f": pf, "v": v.flatten()[0]})

        ## update wf
        # p = self.get_v_pred(torch.Tensor(self.xs)).numpy().tolist()
        # new_wf = []
        # for ps, f in zip(p, self.fs):
        #     new_wf.append(ps*f)
        # self.wf = new_wf


        return 

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """


        # TODO: enter your code here
        # parameters, values = self.ax_clienlinspt.get_best_parameters()
        # x = parameters['x']

        # Get into evaluation (predictive posterior) mode
        # train the gpv model
        xtrain = torch.tensor(self.xs, dtype=torch.double)
        ytrain = torch.tensor(self.fs, dtype=torch.double)
        # standardize(ytrain)
        noises = torch.ones(len(self.xs))*0.15
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = torch.ones(21) * 0.001)
        gpf = GPF(xtrain, ytrain, likelihood)

        # Find optimal model hyperparameters
        gpf.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(gpf.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpf)

        for i in range(len(self.xs) + 20):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = gpf(xtrain)
            # Calc loss and backprop gradients
            loss = -mll(output, ytrain)
            loss.backward()
            # print('Iter %d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, loss.item(),
            #     self.gpv.covar_module.base_kernel.lengthscale.item(),
            #     self.gpv.likelihood.noise.item()
            # ))
            optimizer.step()
        gpf.eval()
        likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        min_x = np.min(self.xs)
        max_x = np.max(self.xs)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(min_x, max_x, 100)
            observed_pred = likelihood(gpf(test_x), noise = torch.ones(test_x.shape[0])*0.15)
            std = observed_pred.stddev.numpy()
            observed_pred = observed_pred.mean.numpy()
            v_pred = self.get_v_pred(test_x).tolist()

        fhat = np.min(self.fs)
        xhat = self.xs[np.argmin(self.xs)]
        vhat = self.vs[np.argmin(self.vs)]
        test_x = test_x.numpy().tolist()
        # print(f"vpred: {v_pred}")
        for i in range(len(self.xs)):
            if observed_pred[i] > fhat and v_pred[i] > 0.9:
                fhat = observed_pred[i]
                xhat = test_x[i]
                vhat = v_pred[i]
        # print(f"xs: {test_x}, \n vpreds_p {v_pred}, \n fs: {observed_pred}\n \n \n \n \n")
        # print(f" gpv model best f: {fhat} \n with p: {vhat}, \n x: {xhat} \n\ \n")



        p = self.get_v_pred(torch.Tensor(self.xs)).tolist()
        best_f = 0
        best_x = 0
        best_v = 0
        index = 0
        
        for i in range(len(self.xs)):
            if self.fs[i] > best_f and self.vs[i] > 1.2 and p[i] > 0.9:
                best_f = self.fs[i]
                best_x = self.xs[i]
                best_v = self.vs[i]
                index = i
        # print(f"\n \n \n best_x: {best_x}, \n best_f: {best_f}, \n best_v: {best_v}, p: {p[index]}")
        prob = p[index]
        if best_f < fhat:
            best_x = xhat
            prob = vhat
            best_f = fhat
        if best_v == 0:
            best_x = self.xs[np.argmax(p)]

        # print(f"\n \n \n x: {best_x}, \n f: {best_f}, \n v: {best_v}")
        # print(f"p: {p[index]}, fs: {self.fs}, xs: {self.xs}, vs: {self.vs}")
        return best_x


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
    return np.random.rand()*5

def main():
    # Init problem
    agent = BO_algo()
    print("this is never being run")
    
    # # Add initial safe point
    # x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
    #         1, n_dim)
    # obj_val = f(x_init)
    # cost_val = v(x_init)
    # agent.add_data_point(x_init, obj_val, cost_val)
    
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
