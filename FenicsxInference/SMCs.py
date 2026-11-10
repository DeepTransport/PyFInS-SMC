import copy

import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from BaseModels import SolverBase, SolverAdjointCache, PriorBase, LikelihoodBase
from InferenceTools.TransitionKernels import OneStepKernel
from InferenceTools.Utilities import adapt_temperature, InferenceRNG
from .LikelihoodInformedSubspace import LikelihoodInformedSubspace


class SMC:
    """
    SMC for replicated, non-distributed parameter samples.

    The FEniCSx forward model may run collectively over MPI, but each complete
    parameter sample is replicated on all ranks. Transition kernels, resampling,
    adaptation and accept/reject randomness are performed on root.
    """

    def __init__(self, 
        prior:PriorBase, 
        forward:SolverBase, 
        like:LikelihoodBase, 
        kernel_template:OneStepKernel,
        max_iter=1000, 
        n_samples=2000, 
        n_mutation_steps=10, 
        n_final_mutation_steps=10,
        ess_tol=0.5, 
        min_beta=1e-6, 
        root=0, 
        seed=12345,
    ):
        self.prior = prior
        self.forward = forward
        self.like = like
        self.kernel_template = kernel_template

        self.comm = getattr(forward, "comm", MPI.COMM_WORLD)
        self.root = root
        self.rng = InferenceRNG(seed) if self.comm.rank == self.root else None

        if hasattr(prior, "nV_owned") and hasattr(prior, "nV_global"):
            if self.comm.size > 1 and prior.nV_owned != prior.nV_global:
                raise ValueError("SMC assumes replicated parameter samples. Use LikelihoodInformedSMC for distributed parameters.")

        if kernel_template.need_grad and not isinstance(forward, SolverAdjointCache):
            raise TypeError("Gradient-based kernels require SolverAdjointCache.")

        self.max_iter = max_iter
        self.n_samples = n_samples
        self.n_mutation_steps = n_mutation_steps
        self.n_final_mutation_steps = n_final_mutation_steps
        self.ess_tol = ess_tol
        self.min_beta = min_beta

        self.kernels = []
        self.xi_data = []
        self.like_data = []
        self.log_zs = []
        self.betas = []

    def _model_evaluation(self, xi_sample, need_grad=False):
        n_samples, n_parameter = xi_sample.shape
        like_potential = np.zeros(n_samples)
        like_grad = np.zeros((n_samples, n_parameter)) if need_grad else None

        iterator = tqdm(range(n_samples), desc="model", disable=self.comm.rank != self.root)
        for i in iterator:
            x = self.prior.transform(xi_sample[i])

            if need_grad:
                Y, _ = self.forward.solve_and_cache(x)
                grad_y = self.like.eval_potential_grad(Y)
                grad_x = self.forward.linearized_observation_transpose(grad_y)
                like_grad[i] = self.prior.linearized_transform_transpose(grad_x)
            else:
                Y, _ = self.forward.solve(x)

            like_potential[i] = self.like.eval_potential(Y)[0]

        return like_potential, like_grad

    def _resample(self, weights, n=None):
        if n is None:
            n = len(weights)

        if self.comm.rank == self.root:
            weights = weights/np.sum(weights)
            indices = self.rng.resample.choice(len(weights), size=n, replace=True, p=weights)
        else:
            indices = None

        return self.comm.bcast(indices, root=self.root)

    def mutation(self, beta, xi_init, like_init, like_grad_init, nsteps, kernel):
        if xi_init.ndim == 1:
            xi_init = xi_init[None, :]

        n_samples, n_parameter = xi_init.shape
        xi_data = np.zeros((nsteps, n_samples, n_parameter))
        like_data = np.zeros((nsteps, n_samples))
        mh_data = np.zeros((nsteps, n_samples))
        acc_data = np.zeros((nsteps, n_samples), dtype=bool)
        sigma_data = np.zeros(nsteps)

        xi_curr = xi_init.copy()
        like_curr = like_init.copy()
        potential_curr = beta*like_curr

        if kernel.need_grad:
            grad_curr = like_grad_init.copy()
            potential_grad_curr = beta*grad_curr
        else:
            grad_curr = None
            potential_grad_curr = None

        sigma = kernel.sigma if self.comm.rank == self.root else None
        acc_num = 0
        batch = 0

        iterator = tqdm(range(nsteps), desc="MCMC", disable=self.comm.rank != self.root)
        for iter_idx in iterator:
            if self.comm.rank == self.root:
                xi_next = kernel.propose(xi_curr, potential_curr, potential_grad_curr)
            else:
                xi_next = None
            xi_next = self.comm.bcast(xi_next, root=self.root)

            like_next, grad_next = self._model_evaluation(xi_next, kernel.need_grad)
            potential_next = beta*like_next
            potential_grad_next = beta*grad_next if kernel.need_grad else None

            if self.comm.rank == self.root:
                log_alpha = kernel.log_MH_ratio(potential_next, xi_next, potential_grad_next)
                accepted = np.log(self.rng.kernel.random(n_samples)) < log_alpha
            else:
                log_alpha = None
                accepted = None
            log_alpha = self.comm.bcast(log_alpha, root=self.root)
            accepted = self.comm.bcast(accepted, root=self.root)

            xi_curr[accepted] = xi_next[accepted]
            like_curr[accepted] = like_next[accepted]
            potential_curr[accepted] = potential_next[accepted]

            if kernel.need_grad:
                grad_curr[accepted] = grad_next[accepted]
                potential_grad_curr[accepted] = potential_grad_next[accepted]

            xi_data[iter_idx] = xi_curr
            like_data[iter_idx] = like_curr
            mh_data[iter_idx] = log_alpha
            acc_data[iter_idx] = accepted

            if self.comm.rank == self.root:
                acc_num += np.sum(accepted)

                if kernel.adapt:
                    batch += 1
                    kernel.accumulate(xi_curr)

                    if batch >= kernel.nbatch:
                        delta = min(0.05, np.sqrt(kernel.nbatch/(iter_idx+1)))
                        acc_rate = acc_num/(kernel.nbatch*n_samples)
                        sigma += -delta if acc_rate < kernel.mh_rate else delta
                        kernel.rebuild(sigma)
                        iterator.set_postfix(acc_rate="{:.3f}".format(acc_rate), sigma="{:.5f}".format(sigma), refresh=True)
                        batch = 0
                        acc_num = 0

            sigma_data[iter_idx] = self.comm.bcast(sigma if self.comm.rank == self.root else None, root=self.root)

        kernel_out = kernel if self.comm.rank == self.root else None
        return xi_data, like_data, grad_curr, kernel_out, mh_data, acc_data, sigma_data

    def iterate(self):
        if self.comm.rank == self.root:
            xi_sample = self.prior.random(self.n_samples)
            sigma = self.kernel_template.sigma
        else:
            xi_sample = None
            sigma = None

        xi_sample = self.comm.bcast(xi_sample, root=self.root)
        sigma = self.comm.bcast(sigma, root=self.root)

        need_grad = self.kernel_template.need_grad
        like_potential, like_grad = self._model_evaluation(xi_sample, need_grad)

        for iter_idx in range(self.max_iter):
            beta_previous = self.betas[-1] if iter_idx > 0 else 0.0
            beta, ess2n = adapt_temperature(beta_previous, like_potential, self.ess_tol, self.min_beta)
            self.betas.append(beta)

            log_weight = (beta_previous-beta)*like_potential
            ref_weight = np.max(log_weight)
            log_z = np.log(np.mean(np.exp(log_weight-ref_weight))) + ref_weight
            self.log_zs.append(log_z)

            if self.comm.rank == self.root:
                print("iteration {}, temperature {}, ess {}, log const {}".format(iter_idx, beta, ess2n, log_z), flush=True)

            weights = np.exp(log_weight-ref_weight)
            weights /= np.sum(weights)
            indices = self._resample(weights)

            xi_resampled = xi_sample[indices]
            like_resampled = like_potential[indices]
            like_grad_resampled = like_grad[indices] if need_grad else None

            if self.comm.rank == self.root:
                kernel = copy.deepcopy(self.kernel_template)
                kernel.set_rng(self.rng.kernel)
                kernel.build_from_samples(xi_resampled, sigma)
            else:
                kernel = copy.deepcopy(self.kernel_template)

            nsteps = self.n_final_mutation_steps if abs(beta-1.0) < 1e-10 else self.n_mutation_steps
            xi_mutation, like_mutation, like_grad, kernel_out, *_ = self.mutation(
                beta, xi_resampled, like_resampled, like_grad_resampled, nsteps, kernel
            )

            self.kernels.append(kernel_out)
            self.xi_data.append(xi_mutation)
            self.like_data.append(like_mutation)

            if abs(beta-1.0) < 1e-10:
                break

            xi_sample = xi_mutation[-1]
            like_potential = like_mutation[-1]

            if self.comm.rank == self.root:
                sigma = kernel_out.sigma
            sigma = self.comm.bcast(sigma if self.comm.rank == self.root else None, root=self.root)


class LikelihoodInformedSMC:
    """
    SMC for distributed FEniCSx Gaussian priors.

    Full whitened samples are distributed by owned parameter DOFs:
        xi_sample.shape = (n_samples, prior.nV_owned)

    LIS coefficients are replicated. Transition kernels, resampling,
    adaptation and accept/reject randomness are performed on root.
    """

    def __init__(self, 
        prior:PriorBase, 
        forward:SolverAdjointCache, 
        like:LikelihoodBase, 
        LIS_template:LikelihoodInformedSubspace,
        kernel_template:OneStepKernel, 
        max_iter=1000, 
        n_samples=2000, 
        n_mutation_steps=10,
        n_final_mutation_steps=10, 
        pm_sample_size=5, 
        ess_tol=0.5, 
        min_beta=1e-6,
        root=0, 
        seed=12345,
    ):
        self.prior = prior
        self.forward = forward
        self.like = like
        self.LIS_template = LIS_template
        self.kernel_template = kernel_template

        self.comm = prior.comm
        self.root = root
        self.rng = InferenceRNG(seed) if self.comm.rank == self.root else None

        self.max_iter = max_iter
        self.n_samples = n_samples
        self.n_mutation_steps = n_mutation_steps
        self.n_final_mutation_steps = n_final_mutation_steps
        self.pm_sample_size = pm_sample_size
        self.ess_tol = ess_tol
        self.min_beta = min_beta

        self.LISs = []
        self.kernels = []
        self.xi_data = []
        self.like_data = []
        self.log_zs = []
        self.betas = []

    def _resample(self, weights, n=None):
        if n is None:
            n = len(weights)

        if self.comm.rank == self.root:
            weights = weights/np.sum(weights)
            indices = self.rng.resample.choice(len(weights), size=n, replace=True, p=weights)
        else:
            indices = None

        return self.comm.bcast(indices, root=self.root)

    def _model_evaluation(self, xi_sample):
        n_samples = xi_sample.shape[0]
        like_potential = np.zeros(n_samples)
        like_grad = np.zeros((n_samples, self.prior.nV_owned))

        iterator = tqdm(range(n_samples), desc="model+adjoint", disable=self.comm.rank != self.root)
        for i in iterator:
            x = self.prior.transform(xi_sample[i])
            Y, _ = self.forward.solve_and_cache(x)
            like_potential[i] = self.like.eval_potential(Y)[0]

            grad_y = self.like.eval_potential_grad(Y)
            grad_x = self.forward.linearized_observation_transpose(grad_y)
            like_grad[i] = self.prior.LIS_linear(grad_x)

        return like_potential, like_grad

    def _pm_evaluate(self, xi, need_grad=False):
        x = self.prior.transform(xi)
        if need_grad:
            Y, _ = self.forward.solve_and_cache(x)
            grad_y = self.like.eval_potential_grad(Y)
            grad_x = self.forward.linearized_observation_transpose(grad_y)
            grad = self.prior.linearized_transform_transpose(grad_x)
        else:
            Y, _ = self.forward.solve(x)
            grad = None
        return self.like.eval_potential(Y)[0], grad

    def minus_sub_pm(self, beta, LIS, xi_sub, pm_size, need_grad=False):
        if xi_sub.ndim == 1:
            xi_sub = xi_sub[None, :]

        n_samples = xi_sub.shape[0]
        like_pm = np.zeros(n_samples)
        like_potential = np.zeros(n_samples)
        xi_sample = np.zeros((n_samples, self.prior.nV_owned))
        like_grad_pm = np.zeros((n_samples, LIS.n_sub)) if need_grad else None

        for i in range(n_samples):
            random_sample = self.prior.random(pm_size)
            xi_null, _ = LIS.projection_null(random_sample)
            candidates = LIS.fill_in(xi_sub[i], xi_null)

            potential_tmp = np.zeros(pm_size)
            grad_tmp = np.zeros((pm_size, self.prior.nV_owned)) if need_grad else None

            for j in range(pm_size):
                potential_tmp[j], grad = self._pm_evaluate(candidates[j], need_grad)
                if need_grad:
                    grad_tmp[j] = grad

            ref = np.max(-beta*potential_tmp)
            weights = np.exp(-beta*potential_tmp-ref)
            normalizer = np.sum(weights)
            weights /= normalizer

            index = self._resample(weights, 1)[0]
            like_pm[i] = -np.log(normalizer/pm_size)-ref
            like_potential[i] = potential_tmp[index]
            xi_sample[i] = candidates[index]

            if need_grad:
                like_grad_pm[i] = beta*LIS.projection_coeff(weights@grad_tmp)

        return like_pm, like_grad_pm, like_potential, xi_sample

    def mutation_pm(self, beta, LIS, xi_init, nsteps, kernel):
        if xi_init.ndim == 1:
            xi_init = xi_init[None, :]

        n_samples = xi_init.shape[0]
        xi_data = np.zeros((nsteps, n_samples, self.prior.nV_owned))
        like_data = np.zeros((nsteps, n_samples))
        mh_data = np.zeros((nsteps, n_samples))
        acc_data = np.zeros((nsteps, n_samples), dtype=bool)
        sigma_data = np.zeros(nsteps)

        xi_sub_curr = LIS.projection_coeff(xi_init)
        like_pm_curr, like_grad_pm_curr, like_curr, xi_curr = self.minus_sub_pm(
            beta, LIS, xi_sub_curr, self.pm_sample_size, kernel.need_grad
        )

        sigma = kernel.sigma if self.comm.rank == self.root else None
        acc_num = 0
        batch = 0

        iterator = tqdm(range(nsteps), desc="MCMC-LIS", disable=self.comm.rank != self.root)
        for iter_idx in iterator:
            if self.comm.rank == self.root:
                xi_sub_next = kernel.propose(xi_sub_curr, like_pm_curr, like_grad_pm_curr)
            else:
                xi_sub_next = None
            xi_sub_next = self.comm.bcast(xi_sub_next, root=self.root)

            like_pm_next, like_grad_pm_next, like_next, xi_next = self.minus_sub_pm(
                beta, LIS, xi_sub_next, self.pm_sample_size, kernel.need_grad
            )

            if self.comm.rank == self.root:
                log_alpha = kernel.log_MH_ratio(like_pm_next, xi_sub_next, like_grad_pm_next)
                accepted = np.log(self.rng.kernel.random(n_samples)) < log_alpha
            else:
                log_alpha = None
                accepted = None
            log_alpha = self.comm.bcast(log_alpha, root=self.root)
            accepted = self.comm.bcast(accepted, root=self.root)

            xi_sub_curr[accepted] = xi_sub_next[accepted]
            like_pm_curr[accepted] = like_pm_next[accepted]
            like_curr[accepted] = like_next[accepted]
            xi_curr[accepted] = xi_next[accepted]

            if kernel.need_grad:
                like_grad_pm_curr[accepted] = like_grad_pm_next[accepted]

            xi_data[iter_idx] = xi_curr
            like_data[iter_idx] = like_curr
            mh_data[iter_idx] = log_alpha
            acc_data[iter_idx] = accepted

            if self.comm.rank == self.root:
                acc_num += np.sum(accepted)

                if kernel.adapt:
                    batch += 1
                    kernel.accumulate(xi_sub_curr)

                    if batch >= kernel.nbatch:
                        delta = min(0.05, np.sqrt(kernel.nbatch/(iter_idx+1)))
                        acc_rate = acc_num/(kernel.nbatch*n_samples)
                        sigma += -delta if acc_rate < kernel.mh_rate else delta
                        kernel.rebuild(sigma)
                        iterator.set_postfix(acc_rate="{:.3f}".format(acc_rate), sigma="{:.5f}".format(sigma), refresh=True)
                        batch = 0
                        acc_num = 0

            sigma_data[iter_idx] = self.comm.bcast(sigma if self.comm.rank == self.root else None, root=self.root)

        kernel_out = kernel if self.comm.rank == self.root else None
        return xi_data, like_data, kernel_out, mh_data, acc_data, sigma_data

    def iterate(self):
        sigma = self.kernel_template.sigma if self.comm.rank == self.root else None
        sigma = self.comm.bcast(sigma, root=self.root)
        xi_sample = self.prior.random(self.n_samples)

        for iter_idx in range(self.max_iter):
            like_potential, like_grad = self._model_evaluation(xi_sample)

            beta_previous = self.betas[-1] if iter_idx > 0 else 0.0
            beta, ess2n = adapt_temperature(beta_previous, like_potential, self.ess_tol, self.min_beta)
            self.betas.append(beta)

            log_weight = (beta_previous-beta)*like_potential
            ref_weight = np.max(log_weight)
            log_z = np.log(np.mean(np.exp(log_weight-ref_weight))) + ref_weight
            self.log_zs.append(log_z)

            weights = np.exp(log_weight-ref_weight)
            weights /= np.sum(weights)

            weighted_grad = like_grad*(beta*np.sqrt(weights)[:, None])
            LIS = self.LIS_template.copy_empty()
            LIS.build_from_grads(weighted_grad)
            self.LISs.append(LIS)

            if self.comm.rank == self.root:
                print(
                    "iteration {}, temperature {}, ess {}, log const {}, LIS dimension {}".format(
                        iter_idx, beta, ess2n, log_z, LIS.n_sub
                    ),
                    flush=True,
                )

            indices = self._resample(weights)
            xi_resampled = xi_sample[indices]
            xi_sub_resampled = LIS.projection_coeff(xi_resampled)

            if self.comm.rank == self.root:
                kernel = copy.deepcopy(self.kernel_template)
                kernel.set_rng(self.rng.kernel)
                kernel.build_from_samples(xi_sub_resampled, sigma)
            else:
                kernel = copy.deepcopy(self.kernel_template)

            nsteps = self.n_final_mutation_steps if abs(beta-1.0) < 1e-10 else self.n_mutation_steps
            xi_mutation, like_mutation, kernel_out, *_ = self.mutation_pm(beta, LIS, xi_resampled, nsteps, kernel)

            self.kernels.append(kernel_out)
            self.xi_data.append(xi_mutation)
            self.like_data.append(like_mutation)

            if abs(beta-1.0) < 1e-10:
                break

            xi_sample = xi_mutation[-1]

            if self.comm.rank == self.root:
                sigma = kernel_out.sigma
            sigma = self.comm.bcast(sigma if self.comm.rank == self.root else None, root=self.root)
