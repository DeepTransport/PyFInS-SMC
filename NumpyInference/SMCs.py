import copy
import time

import numpy as np
from tqdm import tqdm

from BaseModels import SolverBase, SolverAdjointCache
from BaseModels import PriorBase
from BaseModels import LikelihoodBase
from InferenceTools.TransitionKernels import OneStepKernel
from InferenceTools.Utilities import adapt_temperature, InferenceRNG

from .LikelihoodInformedSubspace import LikelihoodInformedSubspace


class SMC:
    """Sequential Monte Carlo with adaptive tempering and MCMC mutation."""

    def __init__(
        self,
        prior: PriorBase,
        forward: SolverBase,
        like: LikelihoodBase,
        kernel_template: OneStepKernel,
        *,
        max_iter:int=1000,
        n_samples:int=2000,
        n_mutation_steps:int=10,
        n_final_mutation_steps:int=10,
        ess_tol:float=0.5,
        min_beta:float=1e-6,
        seed=12345,
    ):
        self.prior = prior
        self.forward = forward
        self.like = like

        self.kernel_template = kernel_template

        if self.kernel_template.need_grad and not isinstance(self.forward, SolverAdjointCache):
            raise TypeError("The selected kernel requires gradient information, but the "
                            "forward solver does not implement SolverAdjointCache.")

        self.kernels = []

        self.max_iter = max_iter
        self.n_samples = n_samples
        self.n_mutation_steps = n_mutation_steps
        self.n_final_mutation_steps = n_final_mutation_steps

        self.xi_data = []
        self.like_data = []
        self.log_zs = []

        self.betas = []
        self.ess_tol = ess_tol
        self.min_beta = min_beta

        self.rng = InferenceRNG(seed)

    def _model_evaluation(self, xi_samples, need_grad=False):
        #t0 = time.perf_counter()
        #print("Start model evaluations", flush=True)
        if self.forward.block_mode and need_grad:
            raise NotImplementedError("Gradient evaluation is not implemented in block mode.")
        if self.forward.block_mode:
            x = self.prior.transform(xi_samples)
            y, _ = self.forward.solve(x)
            potentials = self.like.eval_potential(y)
            gradients = None
        else:
            n_samples, param_dim = xi_samples.shape
            potentials = np.zeros(n_samples)
            gradients = np.zeros((n_samples, param_dim)) if need_grad else None
            for i in range(n_samples):
            #for i in tqdm(range(n_samples)):
                x = self.prior.transform(xi_samples[i])
                if need_grad:
                    y, _ = self.forward.solve_and_cache(x)
                    potentials[i] = self.like.eval_potential(y)[0]
                    grad_y = self.like.eval_potential_grad(y)
                    grad_x = self.forward.linearized_observation_transpose(grad_y)
                    gradients[i] = self.prior.linearized_transform_transpose(grad_x)
                else:
                    y, _ = self.forward.solve(x)
                    potentials[i] = self.like.eval_potential(y)[0]
        #elapsed = time.perf_counter() - t0
        #print(f"Total time for model evaluations is {elapsed}", flush=True)
        return potentials, gradients

    def _weighted_resample(self, weights, n=None):
        weights = weights / np.sum(weights)
        if n is None:
            n = len(weights)
        return self.rng.resample.choice(len(weights), size=n, replace=True, p=weights)

    def mutation(self, beta, xi_init, like_init, like_grad_init, nsteps, kernel):
        if xi_init.ndim == 1:
            xi_init = xi_init[None, :]

        n_samples, param_dim = xi_init.shape

        xi_data = np.zeros((nsteps, n_samples, param_dim))
        like_data = np.zeros((nsteps, n_samples))
        mh_data = np.zeros((nsteps, n_samples))
        acc_data = np.zeros((nsteps, n_samples), dtype=bool)
        sigma_data = np.zeros(nsteps)

        sigma = kernel.sigma
        acc_num = 0
        batch = 0

        xi_curr = xi_init.copy()
        like_curr = like_init.copy()

        # like_curr, grad_curr = self._model_evaluation(xi_curr, kernel.need_grad)

        like_potential_curr = beta * like_curr
        if kernel.need_grad:
            like_grad_curr = like_grad_init.copy()
            like_potential_grad_curr = beta * like_grad_curr
        else:
            like_grad_curr = None
            like_potential_grad_curr = None

        pbar = tqdm(range(nsteps), desc="MCMC")
        for iter_idx in pbar:
        #for iter_idx in range(nsteps):
            xi_next = kernel.propose(xi_curr, like_potential_curr, like_potential_grad_curr)
            like_next, like_grad_next = self._model_evaluation(xi_next, kernel.need_grad)
            like_potential_next = beta * like_next
            if kernel.need_grad:
                like_potential_grad_next = beta * like_grad_next
            else:
                like_potential_grad_next = None
            log_alpha = kernel.log_MH_ratio(like_potential_next, xi_next, like_potential_grad_next)
            accepted = np.log(self.rng.kernel.random(n_samples)) < log_alpha

            xi_curr[accepted] = xi_next[accepted]
            like_curr[accepted] = like_next[accepted]
            like_potential_curr[accepted] = like_potential_next[accepted]

            if kernel.need_grad:
                like_grad_curr[accepted] = like_grad_next[accepted]
                like_potential_grad_curr[accepted] = like_potential_grad_next[accepted]

            xi_data[iter_idx] = xi_curr
            like_data[iter_idx] = like_curr
            mh_data[iter_idx] = log_alpha
            acc_data[iter_idx] = accepted
            sigma_data[iter_idx] = sigma

            acc_num += np.sum(accepted)
            if kernel.adapt:
                batch += 1
                kernel.accumulate(xi_curr)
                if batch >= kernel.nbatch:
                    delta = min( 0.05, np.sqrt(kernel.nbatch / (iter_idx+1)) )
                    acc_rate = acc_num / (kernel.nbatch * n_samples)
                    if acc_rate < kernel.mh_rate:
                        sigma -= delta
                    else:
                        sigma += delta
                    kernel.rebuild(sigma)
                    #print(
                    #    f"    MC iter {iter_idx + 1}, "
                    #    f"acceptance rate {acc_rate}, "
                    #    f"sigma {sigma}"
                    #)
                    # Show the newly adapted sigma immediately
                    pbar.set_postfix(
                        acc_rate=f"{acc_rate:.3f}",
                        sigma=f"{sigma:.5f}",
                        refresh=True,
                    )
                    batch = 0
                    acc_num = 0
                else:
                    # Current cumulative acceptance rate within the batch
                    current_acc_rate = acc_num / batch / n_samples
                    pbar.set_postfix(
                        acc_rate=f"{current_acc_rate:.3f}",
                        sigma=f"{sigma:.5f}",
                        refresh=False,
                    )

        return xi_data, like_data, like_grad_curr, kernel, mh_data, acc_data, sigma_data

    def iterate(self):
        sigma = self.kernel_template.sigma
        need_grad = self.kernel_template.need_grad
        xi_samples = self.prior.random(self.n_samples)
        like_potential, like_grad = self._model_evaluation(xi_samples, need_grad=need_grad)

        for iter_idx in range(self.max_iter):
            beta_previous = self.betas[-1] if iter_idx > 0 else 0.0

            print(f"iteration {iter_idx}")
            beta, ess2n = adapt_temperature(beta_previous, like_potential, self.ess_tol, self.min_beta)
            self.betas.append(beta)
            log_weight = (beta_previous - beta) * like_potential
            ref_weight = np.max(log_weight)
            log_z = np.log( np.mean( np.exp(log_weight - ref_weight) ) ) + ref_weight
            self.log_zs.append(log_z)

            print(
                f"temperature {beta}, "
                f"ess {ess2n}, "
                f"log const {log_z}"
            )

            weights = np.exp(log_weight - ref_weight)
            weights /= np.sum(weights)

            indices = self._weighted_resample(weights)
            xi_resampled = xi_samples[indices]
            like_potential_resampled = like_potential[indices]
            if need_grad:
                like_grad_resampled = like_grad[indices]
            else:
                like_grad_resampled = None

            kernel = copy.deepcopy(self.kernel_template)
            kernel.set_rng(self.rng.kernel)
            kernel.build_from_samples(xi_resampled, sigma)

            if abs(beta - 1.0) < 1e-10:
                nsteps = self.n_final_mutation_steps
            else:
                nsteps = self.n_mutation_steps

            xi_mutation, like_mutation, like_grad, kernel, *_ = self.mutation(
                beta,
                xi_resampled,
                like_potential_resampled,
                like_grad_resampled,
                nsteps,
                kernel,
            )

            self.kernels.append(kernel)
            self.xi_data.append(xi_mutation)
            self.like_data.append(like_mutation)

            if abs(beta - 1.0) < 1.0e-10:
                break

            xi_samples = xi_mutation[-1]
            like_potential = like_mutation[-1]
            sigma = kernel.sigma

            print(f"", flush=True)

class LikelihoodInformedSMC:
    """
    SMC with LIS and pseudo-marginal MCMC.
    """

    def __init__(self, 
        prior:PriorBase, 
        forward:SolverAdjointCache,
        like:LikelihoodBase,
        LIS_template:LikelihoodInformedSubspace,
        kernel_template:OneStepKernel,
        *,
        max_iter:int=1000,
        n_samples:int=2000,
        n_mutation_steps:int=10,
        n_final_mutation_steps:int=10,
        pm_sample_size:int=5,
        ess_tol:float=0.5,
        min_beta:float=1e-6,
        seed=12345,
    ):
        self.prior = prior
        self.forward = forward
        self.like = like
        #
        self.LIS_template = LIS_template
        self.kernel_template = kernel_template
        #
        self.LISs = []
        self.kernels = []
        #
        self.max_iter = max_iter
        #
        #self.kernels.append(kernel)
        self.n_mutation_steps = n_mutation_steps
        self.n_final_mutation_steps = n_final_mutation_steps
        self.n_samples = n_samples
        self.pm_sample_size = pm_sample_size
        #
        self.xi_data = []
        self.like_data = []
        self.log_zs = []
        #
        self.betas = []
        self.ess_tol = ess_tol
        self.min_beta = min_beta
        self.rng = InferenceRNG(seed)
    
    def _model_evaluation(self, xi_samples):
        t0 = time.perf_counter()
        print("Start model evaluations", flush=True)
        #if self.forward.block_mode:
            #x = self.prior.transform(xi_samples)
            #Y = self.forward.solve(x)
            #like_potential = self.like.eval_potential(Y)
            #like_potential_grads = None
        #    raise NotImplementedError("Gradient evaluation is not implemented in block mode.")
        #else:
        n_samples, n_parameter = xi_samples.shape
        like_potential = np.zeros(n_samples)
        like_potential_grads = np.zeros((n_samples,n_parameter))
        for i in tqdm(range(n_samples)):
            xi = self.prior.transform(xi_samples[i])
            Yi, _ = self.forward.solve_and_cache(xi)
            like_potential[i] = self.like.eval_potential(Yi)[0]
            dYi = self.like.eval_potential_grad(Yi)
            dXi = self.forward.linearized_observation_transpose(dYi)
            like_potential_grads[i] = self.prior.LIS_linear(dXi)
        print("Total time for model evaluations is {}".format(time.perf_counter() - t0))
        return like_potential, like_potential_grads

    def _weighted_resample(self, weights, n=None):
        weights = weights / np.sum(weights)
        if n is None:
            n = len(weights)
        return self.rng.resample.choice(len(weights), size=n, replace=True, p=weights)

    def mutation_pm(self, beta, LIS:LikelihoodInformedSubspace, v_init, nsteps, kernel):
        if v_init.ndim == 1:
            v_init = v_init[None, :]
        n_samples, n_parameter = v_init.shape
        x_data = np.zeros((nsteps, n_samples, n_parameter))
        like_data = np.zeros((nsteps, n_samples))
        mh_data = np.zeros((nsteps, n_samples))
        acc_data = np.zeros((nsteps, n_samples), dtype=bool)
        sigma_data = np.zeros(nsteps)

        sigma = kernel.sigma
        acc_num = 0
        batch = 0
        v_sub_curr = LIS.projection_coeff(v_init)
        like_potential_pm_curr, like_grad_pm_curr, like_potential_curr, v_curr = self.minus_sub_pm(
            beta, LIS, v_sub_curr, self.pm_sample_size, kernel.need_grad
        )

        pbar = tqdm(range(nsteps), desc="MCMC")
        for iter_idx in pbar:
            v_sub_next = kernel.propose(v_sub_curr, like_potential_pm_curr, like_grad_pm_curr)
            like_potential_pm_next, like_grad_pm_next, like_potential_next, v_next = self.minus_sub_pm(
                beta, LIS, v_sub_next, self.pm_sample_size, kernel.need_grad
            )
            log_alpha = kernel.log_MH_ratio(like_potential_pm_next, v_sub_next, like_grad_pm_next)
            acc = np.log(self.rng.kernel.random(n_samples)) < log_alpha
            v_sub_curr[acc, :] = v_sub_next[acc, :]
            like_potential_pm_curr[acc] = like_potential_pm_next[acc]
            v_curr[acc, :] = v_next[acc, :]
            like_potential_curr[acc] = like_potential_next[acc]
            if kernel.need_grad:
                like_grad_pm_curr[acc, :] = like_grad_pm_next[acc, :]

            x_data[iter_idx, :, :] = v_curr
            like_data[iter_idx, :] = like_potential_curr
            mh_data[iter_idx, :] = log_alpha
            acc_data[iter_idx, :] = acc
            sigma_data[iter_idx] = sigma

            acc_num += np.sum(acc)
            if kernel.adapt:
                batch += 1
                # Update adaptive covariance stats in LIS coordinates
                kernel.accumulate(v_sub_curr)
                if batch >= kernel.nbatch:
                    delta = min(0.05, np.sqrt(kernel.nbatch / (iter_idx + 1)))
                    acc_rate = acc_num / (kernel.nbatch * n_samples)
                    if acc_rate < kernel.mh_rate:
                        sigma -= delta
                    else:
                        sigma += delta
                    kernel.rebuild(sigma)
                    # print(f"    MC iter {iter_idx + 1}, acceptance rate {acc_rate}, sigma {sigma}")
                    pbar.set_postfix(
                        acc_rate=f"{acc_rate:.3f}",
                        sigma=f"{sigma:.5f}",
                        refresh=True,
                    )
                    batch = 0
                    acc_num = 0
                else:
                    # Current cumulative acceptance rate within the batch
                    current_acc_rate = acc_num / batch / n_samples
                    pbar.set_postfix(
                        acc_rate=f"{current_acc_rate:.3f}",
                        sigma=f"{sigma:.5f}",
                        refresh=False,
                    )
        return x_data, like_data, kernel, mh_data, acc_data, sigma_data
        
    def minus_sub_pm(self, beta, LIS:LikelihoodInformedSubspace, v_sub, pm_size, need_grad=False):
        if v_sub.ndim == 1:
            v_sub = v_sub[None, :]
        n_samples = v_sub.shape[0]
        n_dim = LIS.n_dim
        n_sub = LIS.n_sub
        like_potential_pm = np.zeros(n_samples)
        like_potential = np.zeros(n_samples)
        like_potential_tmp = np.zeros(pm_size)
        v = np.zeros((n_samples, n_dim))
        if need_grad:
            like_grads_pm = np.zeros((n_samples, n_sub))
            grads_tmp = np.zeros((pm_size, n_dim))
            #grads = np.zeros((n_samples, n_dim))
        else:
            like_grads_pm = None
        for i in range(n_samples):
            # Draw null-space random directions
            rs = self.prior.random(pm_size)
            v_null,_ = LIS.projection_null(rs)
            vs = LIS.fill_in(v_sub[i,:], v_null)
            for j in range(pm_size):
                xj = self.prior.transform(vs[j])
                if need_grad:
                    Yj, _ = self.forward.solve_and_cache(xj)
                    like_potential_tmp[j] = self.like.eval_potential(Yj)[0]
                    dYj = self.like.eval_potential_grad(Yj)
                    dXj = self.forward.linearized_observation_transpose(dYj)
                    grads_tmp[j] = self.prior.linearized_transform_transpose(dXj)
                else:
                    Yj, _ = self.forward.solve(xj)
                    like_potential_tmp[j] = self.like.eval_potential(Yj)[0]
            ref = np.max(-beta*like_potential_tmp)
            w_raw = np.exp(-beta*like_potential_tmp-ref)
            tmp = np.sum(w_raw)
            weights = w_raw / tmp
            ind = self._weighted_resample(weights, 1)[0]
            like_potential_pm[i] = -np.log(tmp/pm_size)-ref
            v[i, :] = vs[ind, :]
            like_potential[i] = like_potential_tmp[ind]
            if need_grad:
                weighted_grad_full = weights @ grads_tmp
                like_grads_pm[i, :] = beta*(LIS.projection_coeff(weighted_grad_full))
                #grads[i, :] = grads_tmp[ind, :]
        return like_potential_pm, like_grads_pm, like_potential, v

    def iterate(self):
        sigma = self.kernel_template.sigma
        xi_samples = self.prior.random(self.n_samples)
        for iter_idx in range(self.max_iter):
            print(f"iteration {iter_idx}")
            like_potential, like_grads = self._model_evaluation(xi_samples)
            beta_p = self.betas[-1] if iter_idx > 0 else 0.0
            # Adapt temperature
            beta, ess2n = adapt_temperature(beta_p, like_potential, self.ess_tol, self.min_beta)
            self.betas.append(beta)
            # log normalising constant between adjacent layers
            log_weight = (beta_p - beta) * like_potential
            ref_weight = np.max(log_weight)
            log_z = np.log(np.mean(np.exp(log_weight - ref_weight))) + ref_weight
            self.log_zs.append(log_z)

            print(f"temperature {beta}, ess {ess2n}, log const {log_z}")

            # Importance weights
            weights = np.exp(log_weight-ref_weight)
            weights = weights/np.sum(weights)

            # Weighted gradient SVD
            A = like_grads * (beta*np.sqrt(weights)[:,None])
            LIS = copy.deepcopy(self.LIS_template)
            LIS.build_from_grads(A)
            self.LISs.append(LIS)
            print(f"LIS dimension {LIS.n_sub}")

            # Resampling
            ind = self._weighted_resample(weights)
            kernel = copy.deepcopy(self.kernel_template)
            kernel.set_rng(self.rng.kernel)
            kernel.build_from_samples(LIS.projection_coeff(xi_samples[ind,:]), sigma)

            if abs(beta - 1.0) < 1e-10:
                nsteps = self.n_final_mutation_steps
            else:
                nsteps = self.n_mutation_steps
            xi_mutation, like_mutation, kernel, *_ = self.mutation_pm(beta, LIS, xi_samples[ind, :], nsteps, kernel)

            self.kernels.append(kernel)
            self.xi_data.append(xi_mutation)
            self.like_data.append(like_mutation)
        
            if abs(beta - 1.0) < 1e-10:
                break
            xi_samples = xi_mutation[-1,:,:]
            sigma = kernel.sigma

            print(f"", flush=True)
