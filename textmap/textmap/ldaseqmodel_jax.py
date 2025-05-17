#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# JAX-based implementation inspired by the original Gensim LdaSeqModel.

"""
LdaSeqModel (JAX version)

This module provides a JAX-based implementation of the Dynamic Topic Model (DTM),
focusing on replacing Numba-jitted functions and the `update_obs` optimization
with JAX equivalents for potential performance improvements on accelerators.
The E-step still relies on Gensim's LdaModel and LdaPost for document inference.
"""

import logging
from typing import Tuple, List, Any, Optional

import numpy as np  # Still used for initial setup and some non-JAX parts
import jax
import jax.numpy as jnp
import optax
from scipy.special import gammaln  # For LdaPost, if used, or bound calculations
import tqdm  # For progress bars

from gensim import utils, matutils
from gensim.models import ldamodel  # May still be used for E-step components

# Assuming LdaPost can be imported from the original ldaseqmodel or a utility module
# If LdaPost is in the same directory in ldaseqmodel.py:
from .ldaseqmodel import LdaPost


logger = logging.getLogger(__name__)

# JAX-specific global constants (if any, prefer passing as args or class members)
INIT_VARIANCE_CONST_JAX = 1000.0
INIT_MULT_JAX = 1000.0
OBS_NORM_CUTOFF_JAX = 2.0  # Default, can be instance variable
DEFAULT_JAX_OBS_OPTIMIZER_LR = 0.01
DEFAULT_JAX_OBS_OPTIMIZER_STEPS = 50
LDASQE_EM_THRESHOLD_JAX = 1e-4


# Define the original function
def _jax_compute_post_mean_scan_uncompiled(
    obs_word: jnp.ndarray,  # (T)
    fwd_variance_word: jnp.ndarray,  # (T+1)
    chain_variance: float,
    obs_variance: float,
    num_time_slices: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # mean_word_out (T+1), fwd_mean_word_out (T+1)
    T = num_time_slices
    dtype = obs_word.dtype

    # --- Forward pass for fwd_mean using scan ---
    def fwd_scan_body(carry_fwd_mean_prev, scan_inputs_t_minus_1):
        obs_val_t_minus_1, fwd_var_val_t_minus_1 = scan_inputs_t_minus_1

        denominator = fwd_var_val_t_minus_1 + chain_variance + obs_variance
        # Add epsilon to denominator to prevent 0/0 -> NaN if all terms are 0
        c_factor = jnp.where(
            denominator != 0.0, obs_variance / (denominator + 1e-9), 0.0
        )

        fwd_mean_curr = (
            c_factor * carry_fwd_mean_prev + (1.0 - c_factor) * obs_val_t_minus_1
        )
        return fwd_mean_curr, fwd_mean_curr

    initial_fwd_mean_carry = 0.0
    scan_fwd_inputs = (obs_word, fwd_variance_word[:-1])

    _, fwd_mean_values_1_to_T = jax.lax.scan(
        fwd_scan_body, initial_fwd_mean_carry, scan_fwd_inputs
    )

    fwd_mean_word_out = jnp.zeros(T + 1, dtype=dtype).at[1:].set(fwd_mean_values_1_to_T)

    # --- Backward pass for smooth_mean using scan ---
    def bwd_scan_body(carry_mean_next, scan_inputs_t):
        fwd_mean_val_t, fwd_var_val_t = scan_inputs_t

        fwd_var_t_plus_chain_var = fwd_var_val_t + chain_variance
        # Add epsilon to denominator
        c_factor_backward = jnp.where(
            (chain_variance != 0.0) & (fwd_var_t_plus_chain_var != 0.0),
            chain_variance / (fwd_var_t_plus_chain_var + 1e-9),
            0.0,
        )

        mean_curr = (
            c_factor_backward * fwd_mean_val_t
            + (1.0 - c_factor_backward) * carry_mean_next
        )
        return mean_curr, mean_curr

    initial_bwd_mean_carry = fwd_mean_word_out[T]
    scan_bwd_inputs = (
        jnp.flip(fwd_mean_word_out[:-1]),
        jnp.flip(fwd_variance_word[:-1]),
    )

    _, mean_values_T_minus_1_to_0_rev = jax.lax.scan(
        bwd_scan_body, initial_bwd_mean_carry, scan_bwd_inputs
    )

    mean_word_out = jnp.zeros(T + 1, dtype=dtype)
    mean_word_out = mean_word_out.at[T].set(fwd_mean_word_out[T])
    mean_word_out = mean_word_out.at[:T].set(jnp.flip(mean_values_T_minus_1_to_0_rev))

    return mean_word_out, fwd_mean_word_out

# JIT compile the function
jax_compute_post_mean_scan = jax.jit(_jax_compute_post_mean_scan_uncompiled, static_argnums=(4,))


# Define the original function
def _jax_compute_post_variance_scan_uncompiled(
    obs_variance_scalar: jnp.ndarray,  # JAX scalar
    chain_variance_scalar: jnp.ndarray,  # JAX scalar
    num_time_slices: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # variance_out (T+1), fwd_variance_out (T+1)

    T = num_time_slices # T is now static for this compilation
    # Use the dtype from input JAX scalars
    dtype = obs_variance_scalar.dtype

    # Initialize constants with the right dtype
    _INIT_VARIANCE_CONST_JAX = jnp.array(INIT_VARIANCE_CONST_JAX, dtype=dtype)
    _epsilon = jnp.array(1e-9, dtype=dtype)
    _one = jnp.array(1.0, dtype=dtype)
    _zero = jnp.array(0.0, dtype=dtype)

    # --- Forward pass for fwd_variance ---
    def fwd_var_scan_body(carry_fwd_var_prev, _):  # Loop T times
        denominator = carry_fwd_var_prev + chain_variance_scalar + obs_variance_scalar
        c_factor = jnp.where(
            denominator != _zero, obs_variance_scalar / (denominator + _epsilon), _zero
        )
        fwd_var_curr = c_factor * (carry_fwd_var_prev + chain_variance_scalar)
        return fwd_var_curr, fwd_var_curr

    initial_fwd_var_carry = chain_variance_scalar * _INIT_VARIANCE_CONST_JAX
    _, fwd_var_values_1_to_T = jax.lax.scan(
        fwd_var_scan_body, initial_fwd_var_carry, jnp.arange(num_time_slices)
    )

    fwd_variance_out = jnp.zeros(num_time_slices + 1, dtype=dtype) # Use num_time_slices directly as it's static
    fwd_variance_out = fwd_variance_out.at[0].set(initial_fwd_var_carry)
    fwd_variance_out = fwd_variance_out.at[1:].set(fwd_var_values_1_to_T)

    # --- Backward pass for variance ---
    def bwd_var_scan_body(carry_var_next, fwd_var_val_t):
        c_bwd_factor_den = (
            fwd_var_val_t + chain_variance_scalar
        )  # Use input JAX scalar directly
        c_bwd_ratio = jnp.where(
            c_bwd_factor_den != _zero,
            fwd_var_val_t / (c_bwd_factor_den + _epsilon),
            _zero,
        )
        c_bwd_sq = jnp.power(c_bwd_ratio, 2)
        c_bwd_sq = jnp.where(
            fwd_var_val_t > _epsilon, c_bwd_sq, _zero
        )  # if fwd_var_val_t is ~0, c_bwd_sq is 0

        var_curr = (
            c_bwd_sq * (carry_var_next - chain_variance_scalar)
        ) + (  # Use input JAX scalar directly
            (_one - c_bwd_sq) * fwd_var_val_t
        )
        return var_curr, var_curr

    initial_bwd_var_carry = fwd_variance_out[num_time_slices] # Use num_time_slices directly
    scan_bwd_inputs_var = jnp.flip(fwd_variance_out[:-1])
    _, var_values_T_minus_1_to_0_rev = jax.lax.scan(
        bwd_var_scan_body, initial_bwd_var_carry, scan_bwd_inputs_var
    )

    variance_out = jnp.zeros(num_time_slices + 1, dtype=dtype) # Use num_time_slices directly
    variance_out = variance_out.at[num_time_slices].set(fwd_variance_out[num_time_slices]) # Use num_time_slices
    variance_out = variance_out.at[:num_time_slices].set(jnp.flip(var_values_T_minus_1_to_0_rev)) # Use num_time_slices

    return variance_out, fwd_variance_out

# JIT compile the function
jax_compute_post_variance_scan = jax.jit(_jax_compute_post_variance_scan_uncompiled, static_argnums=(2,))


def jax_objective_for_word_obs(
    x_obs_w: jnp.ndarray,  # (T) - Current observations for the word (variable for optimization)
    # Static parameters for this word, passed as args:
    word_counts_w: jnp.ndarray,  # (T)
    totals_time: jnp.ndarray,  # (T)
    variance_word_fixed: jnp.ndarray,  # (T+1)
    fwd_variance_word_fixed: jnp.ndarray,  # (T+1)
    chain_variance: float,
    obs_variance_scalar: float,
    num_time_slices: int,
    zeta_topic_fixed: jnp.ndarray,  # (T)
) -> float:
    T = num_time_slices
    current_objective_val = 0.0

    mean_w, _ = jax_compute_post_mean_scan(
        x_obs_w,
        fwd_variance_word_fixed,
        chain_variance,
        obs_variance_scalar,
        num_time_slices,
    )

    # Objective terms loop
    def objective_loop_body(t_idx, current_obj_val_accum):
        m_wt_plus_1 = mean_w[t_idx + 1]
        m_wt = mean_w[t_idx]

        term_val = current_obj_val_accum + word_counts_w[t_idx] * m_wt_plus_1

        safe_zeta_t = zeta_topic_fixed[t_idx] + 1e-9
        term_val -= (
            totals_time[t_idx]
            * jnp.exp(m_wt_plus_1 + variance_word_fixed[t_idx + 1] / 2.0)
            / safe_zeta_t
        )

        safe_chain_variance = chain_variance + 1e-9
        chain_term_penalty = (
            (jnp.power(m_wt_plus_1 - m_wt, 2) / (2 * safe_chain_variance))
            - (variance_word_fixed[t_idx + 1] / safe_chain_variance)
            - jnp.log(safe_chain_variance)
        )
        term_val -= chain_term_penalty
        return term_val

    current_objective_val = jax.lax.fori_loop(
        0, T, objective_loop_body, current_objective_val
    )

    # Prior term for m_w[0]
    safe_chain_variance_prior = chain_variance + 1e-9
    safe_init_mult_jax = INIT_MULT_JAX + 1e-9  # Ensure INIT_MULT_JAX is positive
    current_objective_val -= jnp.power(mean_w[0], 2) / (
        2 * safe_init_mult_jax * safe_chain_variance_prior
    )

    # Optimizers minimize, original f_obs returned -(objective for maximization)
    # So, if jax_objective_for_word_obs calculates the positive objective, return -objective
    return -current_objective_val


# Pre-compile the gradient function
# Note: JITting jax.grad(jax_objective_for_word_obs) directly.
# argnums=0 means differentiate w.r.t. x_obs_w
# num_time_slices (index 7 in jax_objective_for_word_obs) must be static.
grad_fn_jax_objective_for_word_obs = jax.jit(
    jax.grad(jax_objective_for_word_obs, argnums=0), static_argnums=(7,)
)


class sslm_jax(utils.SaveLoad):
    """
    JAX-based State Space Language Model for DTM.
    Corresponds to the `sslm` class in the original LdaSeqModel.
    """

    def __init__(
        self,
        vocab_len: int,
        num_time_slices: int,
        num_topics: int,
        obs_variance: float = 0.5,
        chain_variance: float = 0.005,
        obs_optimizer_lr: float = DEFAULT_JAX_OBS_OPTIMIZER_LR,
        obs_optimizer_steps: int = DEFAULT_JAX_OBS_OPTIMIZER_STEPS,
        dtype: jnp.dtype = jnp.float64,
    ):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.num_topics = num_topics
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.dtype = dtype

        self.obs_optimizer_lr = obs_optimizer_lr
        self.obs_optimizer_steps = obs_optimizer_steps
        self.obs_norm_cutoff = OBS_NORM_CUTOFF_JAX

        self.obs = jnp.zeros((vocab_len, num_time_slices), dtype=dtype)
        self.e_log_prob = jnp.zeros((vocab_len, num_time_slices), dtype=dtype)
        self.mean = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.fwd_mean = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.variance = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.fwd_variance = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.zeta = jnp.zeros(num_time_slices, dtype=dtype)

    def update_zeta_jax(self) -> jnp.ndarray:
        exp_terms = jnp.exp(self.mean[:, 1:] + self.variance[:, 1:] / 2.0)
        new_zeta = jnp.sum(exp_terms, axis=0)
        return new_zeta

    def compute_expected_log_prob_jax(self) -> jnp.ndarray:
        log_zeta_broadcast = jnp.log(self.zeta[jnp.newaxis, :] + 1e-9)
        new_e_log_prob = self.mean[:, 1:] - log_zeta_broadcast
        return new_e_log_prob

    def sslm_counts_init_jax(self, sstats_topic: np.ndarray):
        W = self.vocab_len
        T = self.num_time_slices

        log_norm_counts = np.copy(sstats_topic)
        log_norm_counts_sum = np.sum(log_norm_counts)
        if log_norm_counts_sum > 0:
            log_norm_counts /= log_norm_counts_sum
        log_norm_counts += 1.0 / W  # Smoothing
        log_norm_counts_sum_smoothed = np.sum(log_norm_counts)
        if log_norm_counts_sum_smoothed > 0:
            log_norm_counts /= log_norm_counts_sum_smoothed
        log_norm_counts = np.log(log_norm_counts + 1e-9)  # Add epsilon before log

        self.obs = jnp.array(
            np.tile(log_norm_counts[:, np.newaxis], (1, T)), dtype=self.dtype
        )

        # Compute post variance (once, then tile as it's word-independent)
        # Cast scalar Python floats to JAX scalars of the correct dtype
        obs_var_scalar_jax = jnp.array(self.obs_variance, dtype=self.dtype)
        chain_var_scalar_jax = jnp.array(self.chain_variance, dtype=self.dtype)
        single_variance_out, single_fwd_variance_out = jax_compute_post_variance_scan(
            obs_var_scalar_jax, chain_var_scalar_jax, T
        )
        self.variance = jnp.tile(single_variance_out[jnp.newaxis, :], (W, 1))
        self.fwd_variance = jnp.tile(single_fwd_variance_out[jnp.newaxis, :], (W, 1))

        # Compute post mean (vmapped as it depends on self.obs and self.fwd_variance)
        vmap_compute_mean = jax.vmap(
            jax_compute_post_mean_scan, in_axes=(0, 0, None, None, None), out_axes=0
        )
        all_means, all_fwd_means = vmap_compute_mean(
            self.obs, self.fwd_variance, self.chain_variance, self.obs_variance, T
        )
        self.mean = all_means
        self.fwd_mean = all_fwd_means

        self.zeta = self.update_zeta_jax()
        self.e_log_prob = self.compute_expected_log_prob_jax()

    def update_obs_jax(
        self, sstats_topic_all_times: jnp.ndarray, totals_all_times: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        W = self.vocab_len
        T = self.num_time_slices
        optimizer = optax.adam(learning_rate=self.obs_optimizer_lr)

        new_obs_list = []
        norm_cutoff_obs_cache = None

        for w_idx in range(W):
            word_counts_w = sstats_topic_all_times[w_idx, :]
            counts_norm = jnp.linalg.norm(word_counts_w)

            initial_obs_w_val = self.obs[w_idx, :]
            current_obs_w_optimized = initial_obs_w_val

            run_optimization = True
            word_counts_w_for_opt = word_counts_w

            if counts_norm < self.obs_norm_cutoff:
                if norm_cutoff_obs_cache is not None:
                    current_obs_w_optimized = jnp.copy(norm_cutoff_obs_cache)
                    run_optimization = False
                else:
                    word_counts_w_for_opt = jnp.zeros_like(word_counts_w)

            if run_optimization:
                static_args_for_opt_tuple = (
                    word_counts_w_for_opt,
                    totals_all_times,
                    self.variance[w_idx, :],
                    self.fwd_variance[w_idx, :],
                    self.chain_variance,
                    self.obs_variance,
                    T,
                    self.zeta,  # Pass current zeta for this topic
                )

                @jax.jit
                def optimize_single_obs_w_loop(initial_obs_w, *args_tuple_for_loss):
                    opt_state = optimizer.init(initial_obs_w)

                    def opt_step(_, state_tuple):
                        current_obs, current_opt_state = state_tuple
                        grads = grad_fn_jax_objective_for_word_obs(
                            current_obs, *args_tuple_for_loss
                        )
                        updates, new_opt_state = optimizer.apply_updates(
                            grads, current_opt_state, current_obs
                        )
                        new_obs = optax.apply_updates(current_obs, updates)
                        return new_obs, new_opt_state

                    final_obs_w, _ = jax.lax.fori_loop(
                        0,
                        self.obs_optimizer_steps,
                        opt_step,
                        (initial_obs_w, opt_state),
                    )
                    return final_obs_w

                current_obs_w_optimized = optimize_single_obs_w_loop(
                    initial_obs_w_val, *static_args_for_opt_tuple
                )

                if (
                    counts_norm < self.obs_norm_cutoff and run_optimization
                ):  # Cache only if it was just computed
                    norm_cutoff_obs_cache = jnp.copy(current_obs_w_optimized)

            new_obs_list.append(current_obs_w_optimized)

        self.obs = jnp.stack(new_obs_list, axis=0)

        vmap_compute_mean = jax.vmap(
            jax_compute_post_mean_scan, in_axes=(0, 0, None, None, None), out_axes=0
        )
        all_means, all_fwd_means = vmap_compute_mean(
            self.obs, self.fwd_variance, self.chain_variance, self.obs_variance, T
        )
        self.mean = all_means
        self.fwd_mean = all_fwd_means

        self.zeta = self.update_zeta_jax()
        return self.obs, self.zeta

    def compute_bound_jax(
        self, sstats_topic_all_times: jnp.ndarray, totals_all_times: jnp.ndarray
    ) -> float:
        W = self.vocab_len
        T = self.num_time_slices
        safe_chain_variance = self.chain_variance + 1e-9

        bound_val = jnp.sum(self.variance[:, 0] - self.variance[:, T]) / (
            2 * safe_chain_variance
        )

        def bound_time_loop_body(t_loop_idx, current_bound_accum):
            m_t_plus_1 = self.mean[:, t_loop_idx + 1]
            m_t = self.mean[:, t_loop_idx]
            v_t_plus_1 = self.variance[:, t_loop_idx + 1]
            sstats_w_t = sstats_topic_all_times[:, t_loop_idx]

            term1_w_contrib = (
                (jnp.power(m_t_plus_1 - m_t, 2) / (2 * safe_chain_variance))
                - (v_t_plus_1 / safe_chain_variance)
                - jnp.log(safe_chain_variance)
            )
            sum_term1_w = jnp.sum(term1_w_contrib)

            term2_w_contrib = sstats_w_t * m_t_plus_1
            sum_term2_w = jnp.sum(term2_w_contrib)

            ent_w_contrib = jnp.log(jnp.abs(v_t_plus_1) + 1e-9) / 2.0
            sum_ent_w = jnp.sum(ent_w_contrib)

            term3_scalar = -totals_all_times[t_loop_idx] * jnp.log(
                self.zeta[t_loop_idx] + 1e-9
            )

            current_bound_accum += sum_term2_w + term3_scalar + sum_ent_w - sum_term1_w
            return current_bound_accum

        bound_val = jax.lax.fori_loop(0, T, bound_time_loop_body, bound_val)
        return bound_val

    def fit_sslm_jax(
        self,
        sstats_topic_all_times: jnp.ndarray,
        totals_all_times: jnp.ndarray,
        sslm_fit_threshold: float = 1e-6,
        sslm_max_iter: int = 2,
    ) -> float:
        W = self.vocab_len
        T = self.num_time_slices

        # Compute post variance (once, then tile as it's word-independent)
        # Cast scalar Python floats to JAX scalars of the correct dtype
        obs_var_scalar_jax = jnp.array(self.obs_variance, dtype=self.dtype)
        chain_var_scalar_jax = jnp.array(self.chain_variance, dtype=self.dtype)
        single_variance_out, single_fwd_variance_out = jax_compute_post_variance_scan(
            obs_var_scalar_jax, chain_var_scalar_jax, T
        )
        self.variance = jnp.tile(single_variance_out[jnp.newaxis, :], (W, 1))
        self.fwd_variance = jnp.tile(single_fwd_variance_out[jnp.newaxis, :], (W, 1))

        vmap_compute_mean = jax.vmap(
            jax_compute_post_mean_scan, in_axes=(0, 0, None, None, None), out_axes=0
        )
        current_means, current_fwd_means = vmap_compute_mean(
            self.obs, self.fwd_variance, self.chain_variance, self.obs_variance, T
        )
        self.mean = current_means
        self.fwd_mean = current_fwd_means
        self.zeta = self.update_zeta_jax()

        bound = self.compute_bound_jax(sstats_topic_all_times, totals_all_times)
        logger.info("Initial sslm_jax bound is %f", bound)

        for iter_num in range(sslm_max_iter):
            old_bound = bound

            self.obs, self.zeta = self.update_obs_jax(
                sstats_topic_all_times, totals_all_times
            )

            bound = self.compute_bound_jax(sstats_topic_all_times, totals_all_times)

            convergence = jnp.fabs((bound - old_bound) / (old_bound + 1e-9))
            logger.info(
                "sslm_jax iter %i, bound %f, convergence %f",
                iter_num + 1,
                bound,
                convergence,
            )
            if convergence < sslm_fit_threshold:
                break

        self.e_log_prob = self.compute_expected_log_prob_jax()
        return bound


class LdaSeqModelJax(utils.SaveLoad):
    """
    JAX-based LdaSeqModel (Dynamic Topic Model).
    M-step (topic evolution) is JAX-based. E-step (document inference) uses Gensim's LdaModel.
    """

    def __init__(
        self,
        corpus=None,
        time_slice: Optional[List[int]] = None,
        id2word: Optional[Any] = None,
        num_topics: int = 10,
        alphas: float = 0.01,
        chain_variance: float = 0.005,
        obs_variance: float = 0.5,
        random_state_seed: int = 0,
        dtype: jnp.dtype = jnp.float64,
        sslm_obs_optimizer_lr: float = DEFAULT_JAX_OBS_OPTIMIZER_LR,
        sslm_obs_optimizer_steps: int = DEFAULT_JAX_OBS_OPTIMIZER_STEPS,
        lda_inference_max_iter: int = 25,
        em_min_iter: int = 6,
        em_max_iter: int = 20,
        chunksize: int = 100,
        passes: int = 10,
        sslm_max_iter_per_topic: int = 2,  # Max iterations for sslm_jax.fit_sslm_jax
        sslm_convergence_threshold: float = 1e-6,  # Convergence for sslm_jax.fit_sslm_jax
    ):
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError("Need corpus or id2word for vocab size.")

        self.vocab_len = 0
        if self.id2word:
            self.vocab_len = len(self.id2word)
        elif corpus is not None:  # Infer from corpus if id2word not provided
            logger.warning(
                "No id2word provided, inferring from corpus. This may be slow."
            )
            self.id2word = utils.dict_from_corpus(corpus)
            self.vocab_len = len(self.id2word)

        self.num_topics = num_topics
        self.time_slice = time_slice
        if self.time_slice is None and corpus is not None:
            raise ValueError("time_slice must be provided if corpus is given.")
        self.num_time_slices = len(time_slice) if time_slice else 0

        self.alphas_np = np.full(num_topics, alphas, dtype=np.float64)
        self.dtype = dtype
        self.key = jax.random.PRNGKey(random_state_seed)
        self.random_state_np = np.random.RandomState(
            random_state_seed
        )  # For Gensim LDA

        self.lda_inference_max_iter = lda_inference_max_iter
        self.em_min_iter = em_min_iter
        self.em_max_iter = em_max_iter
        self.chunksize = chunksize
        self.sslm_max_iter_per_topic = sslm_max_iter_per_topic
        self.sslm_convergence_threshold = sslm_convergence_threshold

        self.corpus_len = 0
        if corpus is not None:
            try:
                self.corpus_len = len(corpus)  # type: ignore
            except TypeError:
                logger.warning(
                    "Input corpus stream has no len(); counting documents for progress bar."
                )
                self.corpus_len = sum(1 for _ in corpus)  # type: ignore

        self.max_doc_len = 0
        if corpus is not None:
            # This can be slow for a true stream. Consider making it optional or handled differently.
            logger.info("Calculating max document length from corpus...")
            self.max_doc_len = max(len(doc) for doc in corpus) if self.corpus_len > 0 else 0  # type: ignore

        self.topic_chains_jax: List[sslm_jax] = []
        for _ in range(num_topics):
            sslm_instance = sslm_jax(
                vocab_len=self.vocab_len,
                num_time_slices=self.num_time_slices,
                num_topics=self.num_topics,
                chain_variance=chain_variance,
                obs_variance=obs_variance,
                obs_optimizer_lr=sslm_obs_optimizer_lr,
                obs_optimizer_steps=sslm_obs_optimizer_steps,
                dtype=self.dtype,
            )
            self.topic_chains_jax.append(sslm_instance)

        self.gammas_np: Optional[np.ndarray] = None

        if corpus is not None and time_slice is not None:
            logger.info("Initializing LdaSeqModelJax with a standard LDA model...")
            lda_model_init = ldamodel.LdaModel(
                corpus,
                id2word=self.id2word,
                num_topics=self.num_topics,
                passes=passes,
                alpha=self.alphas_np,
                random_state=self.random_state_np,
                dtype=np.float64,
            )
            initial_sstats_beta = np.transpose(lda_model_init.state.sstats)

            self.init_ldaseq_ss_jax(initial_sstats_beta)
            logger.info("LdaSeqModelJax initialized. Starting EM training...")
            self.fit_lda_seq_jax(corpus)

    def init_ldaseq_ss_jax(self, init_suffstats_beta: np.ndarray):
        logger.info("Initializing JAX SSLM topic chains...")
        for k, chain_jax in enumerate(self.topic_chains_jax):
            sstats_for_topic_k = init_suffstats_beta[:, k]
            chain_jax.sslm_counts_init_jax(sstats_for_topic_k)
        logger.info("JAX SSLM topic chains initialized.")

    def make_lda_seq_slice_jax(
        self, lda_model_instance: ldamodel.LdaModel, time_slice_idx: int
    ) -> ldamodel.LdaModel:
        """Updates the LDA model's topics with e_log_prob from JAX chains for a specific time slice."""
        for k_topic in range(self.num_topics):
            # Convert JAX array to NumPy for Gensim LdaModel
            e_log_prob_k_t = np.array(
                self.topic_chains_jax[k_topic].e_log_prob[:, time_slice_idx]
            )
            lda_model_instance.expElogbeta[:, k_topic] = (
                e_log_prob_k_t  # Store log probs
            )
            # Gensim's LdaModel often works with expElogbeta, but if it needs topics directly:
            # lda_model_instance.state.sstats[k_topic, :] = np.exp(e_log_prob_k_t) # Or similar update

        # If LdaModel uses state.get_lambda() or similar, ensure it reflects these values.
        # For simplicity, we assume direct update of expElogbeta is sufficient for inference.
        # If LdaModel.inference uses self.state.sstats or self.state.get_lambda(),
        # those might need to be updated or a custom inference path used.
        # A common way is to set `lda_model_instance.state.expElogbeta`.
        # The original `LdaModel.topics` was a property that returned expElogbeta.
        # Let's assume `lda_model_instance.expElogbeta` is the correct attribute.
        # If not, this part needs adjustment based on how Gensim's LdaModel uses topic-word distributions.
        # A safer bet might be to update `lda_model_instance.state.sstats` if that's what inference uses,
        # or ensure `lda_model_instance.expElogbeta` is correctly used by `LdaPost`.
        # The original `make_lda_seq_slice` updated `lda.topics` which was a property.
        # Let's try to update `lda_model_instance.expElogbeta` as it's often used directly.
        # If `LdaPost` uses `lda.topics[word_id][k]`, this needs to point to the correct log-probabilities.
        # A simple hack for LdaPost if it reads `lda.topics`:
        class TempTopics:
            def __init__(self, exp_e_log_beta_matrix):
                self.exp_e_log_beta_matrix = exp_e_log_beta_matrix

            def __getitem__(self, item):  # word_id, k_topic
                word_id, k_topic_idx = item
                return self.exp_e_log_beta_matrix[word_id, k_topic_idx]

        current_exp_elogbeta = np.zeros(
            (self.vocab_len, self.num_topics), dtype=np.float64
        )
        for k_topic in range(self.num_topics):
            current_exp_elogbeta[:, k_topic] = np.array(
                self.topic_chains_jax[k_topic].e_log_prob[:, time_slice_idx]
            )

        # Instead of setting topics directly (which doesn't exist in LdaModel),
        # set the expElogbeta attribute which is what LdaPost actually uses
        lda_model_instance.expElogbeta = current_exp_elogbeta.T  # Transpose to match LdaModel's expected shape

        lda_model_instance.alpha = np.copy(self.alphas_np)  # Ensure alpha is also set
        return lda_model_instance

    def lda_seq_infer_jax(
        self, corpus, current_em_iter: int
    ) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        """E-step: Perform inference for document variational parameters (gammas)
        and collect sufficient statistics for topics using Gensim's LdaModel and LdaPost.
        """
        bound = 0.0
        # Initialize sufficient statistics as list of NumPy arrays
        topic_suffstats_np = [
            np.zeros((self.vocab_len, self.num_time_slices), dtype=np.float64)
            for _ in range(self.num_topics)
        ]

        # Initialize gammas as NumPy array
        gammas_np = np.zeros((self.corpus_len, self.num_topics), dtype=np.float64)
        # lhoods_np = np.zeros((self.corpus_len, self.num_topics + 1), dtype=np.float64) # If needed by LdaPost

        # Setup Gensim LDA model instance for inference for this E-step
        # We will update its topics for each time slice.
        lda_inference_model = ldamodel.LdaModel(
            num_topics=self.num_topics,
            alpha=self.alphas_np,
            id2word=self.id2word,
            dtype=np.float64,
            # expElogbeta needs to be shaped (num_topics, vocab_len) for Gensim state
            # or ensure LdaPost uses a (vocab_len, num_topics) view.
            # For now, we'll shape it (vocab_len, num_topics) and LdaPost might need adjustment
            # or make_lda_seq_slice_jax handles it.
        )
        # Initialize expElogbeta to avoid issues if not set before first use
        lda_inference_model.expElogbeta = np.zeros(
            (self.num_topics, self.vocab_len), dtype=np.float64
        )

        ldapost_instance = LdaPost(
            max_doc_len=self.max_doc_len,
            num_topics=self.num_topics,
            lda=lda_inference_model,  # Pass the LDA model instance
        )

        doc_idx_overall = 0
        current_time_slice_idx = 0
        docs_in_current_slice = 0

        # Precompute cumulative time slices for easy checking
        cumulative_time_slices = (
            np.cumsum(np.array(self.time_slice))
            if self.time_slice
            else np.array([self.corpus_len])
        )

        for chunk_no, chunk_docs in enumerate(utils.grouper(corpus, self.chunksize)):
            for doc_bow in chunk_docs:
                if doc_bow is None:
                    continue  # Handle potential None from grouper

                # Determine current time slice for the document
                if doc_idx_overall >= cumulative_time_slices[current_time_slice_idx]:
                    current_time_slice_idx += 1
                    docs_in_current_slice = 0  # Reset doc counter for new slice
                    # Update LDA model topics for the new time slice
                    lda_inference_model = self.make_lda_seq_slice_jax(
                        lda_inference_model, current_time_slice_idx
                    )

                # Prepare LdaPost for the current document
                ldapost_instance.doc = doc_bow
                ldapost_instance.gamma = gammas_np[
                    doc_idx_overall
                ]  # Use view for current doc's gamma
                # ldapost_instance.lhood = lhoods_np[doc_idx_overall] # If LdaPost uses this

                # Perform inference for the document
                # The original LdaPost.fit_lda_post took `ldaseq` (self) as an argument,
                # which might be used for callbacks or accessing model parameters.
                # If LdaPost needs access to LdaSeqModelJax instance, pass `self`.
                # For now, assuming it primarily needs `lda_inference_model`.
                doc_lhood = ldapost_instance.fit_lda_post(
                    doc_number=docs_in_current_slice,  # doc index within the current time slice
                    time=current_time_slice_idx,
                    ldaseq=None,  # Or self, if LdaPost expects the main model
                    lda_inference_max_iter=self.lda_inference_max_iter,
                )

                bound += doc_lhood
                gammas_np[doc_idx_overall] = (
                    ldapost_instance.gamma
                )  # Store updated gamma

                # Accumulate sufficient statistics (LdaPost updates topic_suffstats_np in place)
                # LdaPost.update_lda_seq_ss expects topic_suffstats as a list of (V,T) arrays
                topic_suffstats_np = ldapost_instance.update_lda_seq_ss(
                    time=current_time_slice_idx,
                    doc=doc_bow,  # doc_bow is already set in ldapost_instance
                    topic_suffstats=topic_suffstats_np,
                )

                doc_idx_overall += 1
                docs_in_current_slice += 1

        self.gammas_np = gammas_np  # Store final gammas
        return bound, gammas_np, topic_suffstats_np

    def fit_lda_seq_topics_jax(
        self, topic_suffstats_all_topics_jax: List[jnp.ndarray]
    ) -> float:
        """M-step: Fit the SSLM for each topic using JAX."""
        total_topic_bound = 0.0
        for k, chain_jax in enumerate(self.topic_chains_jax):
            logger.info(f"Fitting JAX SSLM for topic {k}...")
            sstats_k_jax = topic_suffstats_all_topics_jax[k]
            totals_k_jax = jnp.sum(sstats_k_jax, axis=0)

            lhood_term = chain_jax.fit_sslm_jax(
                sstats_k_jax,
                totals_k_jax,
                sslm_fit_threshold=self.sslm_convergence_threshold,
                sslm_max_iter=self.sslm_max_iter_per_topic,
            )
            total_topic_bound += lhood_term
        return float(total_topic_bound)  # Ensure float return

    def fit_lda_seq_jax(self, corpus):
        """Main EM training loop for LdaSeqModelJax."""
        logger.info("Starting LdaSeqModelJax EM training...")

        # Constants from original fit_lda_seq
        LOWER_ITER = 10
        ITER_MULT_LOW = 2
        MAX_ITER_INF = 500  # Max inference iterations if bound goes down

        current_lda_inference_max_iter = self.lda_inference_max_iter

        overall_bound = -np.inf  # Initialize with a very small number

        with tqdm.tqdm(total=self.em_max_iter, desc="LdaSeqJax EM Iterations") as pbar:
            for em_iter_num in range(self.em_max_iter):
                pbar.update(1)
                logger.info(f"EM Iteration {em_iter_num + 1}/{self.em_max_iter}")
                old_overall_bound = overall_bound

                # E-step
                logger.info("E-Step started...")
                e_step_bound, _, topic_sstats_np_list = self.lda_seq_infer_jax(
                    corpus, current_em_iter=em_iter_num
                )
                logger.info(f"E-Step finished. Bound from E-step: {e_step_bound}")

                # Convert NumPy sufficient statistics to JAX arrays for M-step
                topic_sstats_jax_list = [
                    jnp.array(ts_np, dtype=self.dtype) for ts_np in topic_sstats_np_list
                ]

                # M-step
                logger.info("M-Step started...")
                m_step_bound = self.fit_lda_seq_topics_jax(topic_sstats_jax_list)
                logger.info(
                    f"M-Step finished. Bound from M-step (topics): {m_step_bound}"
                )

                overall_bound = (
                    e_step_bound + m_step_bound
                )  # Total bound for this EM iteration

                if (
                    overall_bound - old_overall_bound
                ) < 0 and old_overall_bound != -np.inf:
                    logger.warning(
                        f"Bound decreased from {old_overall_bound} to {overall_bound}!"
                    )
                    if current_lda_inference_max_iter < LOWER_ITER:
                        current_lda_inference_max_iter *= ITER_MULT_LOW
                        logger.info(
                            f"Increasing LDA inference iterations to {current_lda_inference_max_iter}"
                        )

                convergence = np.fabs(
                    (overall_bound - old_overall_bound) / (old_overall_bound + 1e-9)
                )
                pbar.set_postfix(
                    {
                        "bound": f"{overall_bound:.4f}",
                        "convergence": f"{convergence:.6f}",
                    }
                )
                logger.info(
                    f"EM Iteration {em_iter_num + 1} finished. "
                    f"Total Bound: {overall_bound:.4f}, Convergence: {convergence:.6f}"
                )

                if (
                    em_iter_num >= self.em_min_iter
                    and convergence < LDASQE_EM_THRESHOLD_JAX
                ):
                    if (
                        current_lda_inference_max_iter >= MAX_ITER_INF
                    ):  # Already at max inf iter
                        logger.info("Convergence threshold reached. Stopping EM.")
                        break
                    else:  # Increase inference iterations for final passes
                        logger.info(
                            f"Convergence near, increasing inference iterations to {MAX_ITER_INF} for final passes."
                        )
                        current_lda_inference_max_iter = MAX_ITER_INF
                        # Reset convergence to ensure at least one more iteration with max_inf_iter
                        # Or, add a flag to indicate "final passes mode"
                        # For simplicity, we'll let it run up to em_max_iter if threshold met early
                        # and inference_max_iter was increased.
            else:  # Loop finished without break
                if (
                    em_iter_num < self.em_max_iter - 1
                ):  # Did not complete all em_max_iter
                    logger.info(
                        "EM loop finished due to reaching convergence criterion."
                    )
                else:  # Completed all em_max_iter
                    logger.info(
                        f"EM loop finished after {self.em_max_iter} iterations."
                    )
        return overall_bound

    def print_topic_times_jax(
        self, topic_idx: int, top_terms: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """Print topic evolution over time for a given topic_idx."""
        if not self.id2word:
            logger.warning("id2word not available, cannot print topics with words.")
            return []
        if not (0 <= topic_idx < self.num_topics):
            logger.error(
                f"Invalid topic_idx {topic_idx}. Must be between 0 and {self.num_topics-1}."
            )
            return []

        chain = self.topic_chains_jax[topic_idx]
        e_log_prob_np = np.array(chain.e_log_prob)

        topics_over_time = []
        for t in range(self.num_time_slices):
            log_probs_t = e_log_prob_np[:, t]
            # Probs for display, not necessarily the same as model's internal representation if using log-probs
            probs_t = np.exp(log_probs_t - np.max(log_probs_t))  # Softmax for stability
            probs_t /= np.sum(probs_t)

            best_n_indices = np.argsort(probs_t)[::-1][:top_terms]
            topic_terms = [
                (self.id2word[idx], float(probs_t[idx]))
                for idx in best_n_indices
                if idx in self.id2word
            ]
            topics_over_time.append(topic_terms)
        return topics_over_time

    def print_topics_jax(
        self, time_slice_idx: int = 0, top_terms: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """Print topics for a specific time slice."""
        if not (0 <= time_slice_idx < self.num_time_slices):
            logger.error(
                f"Invalid time_slice_idx {time_slice_idx}. Must be between 0 and {self.num_time_slices-1}."
            )
            return [[] for _ in range(self.num_topics)]

        all_topics_at_time = []
        for topic_idx in range(self.num_topics):
            topic_description_at_time = self.print_topic_jax(
                topic_idx, time_slice_idx, top_terms
            )
            all_topics_at_time.append(topic_description_at_time)
        return all_topics_at_time

    def print_topic_jax(
        self, topic_idx: int, time_slice_idx: int = 0, top_terms: int = 10
    ) -> List[Tuple[str, float]]:
        """Print a single topic for a specific time slice."""
        if not self.id2word:
            logger.warning("id2word not available, cannot print topic with words.")
            return []
        if not (0 <= topic_idx < self.num_topics):
            logger.error(f"Invalid topic_idx {topic_idx}.")
            return []
        if not (0 <= time_slice_idx < self.num_time_slices):
            logger.error(f"Invalid time_slice_idx {time_slice_idx}.")
            return []

        chain = self.topic_chains_jax[topic_idx]
        e_log_prob_topic_time = np.array(chain.e_log_prob[:, time_slice_idx])

        probs_topic_time = np.exp(e_log_prob_topic_time - np.max(e_log_prob_topic_time))
        probs_topic_time /= np.sum(probs_topic_time)

        best_n_indices = np.argsort(probs_topic_time)[::-1][:top_terms]
        topic_terms = [
            (self.id2word[idx], float(probs_topic_time[idx]))
            for idx in best_n_indices
            if idx in self.id2word
        ]
        return topic_terms

    def __getitem__(
        self,
        doc_bow: List[Tuple[int, int]],
        lda_inference_max_iter: Optional[int] = None,
    ) -> np.ndarray:
        """Get topic distribution for a new document."""
        if lda_inference_max_iter is None:
            lda_inference_max_iter = self.lda_inference_max_iter

        # Setup a temporary LDA model for inference
        lda_inf_model = ldamodel.LdaModel(
            num_topics=self.num_topics,
            alpha=self.alphas_np,
            id2word=self.id2word,
            dtype=np.float64,
        )
        lda_inf_model.expElogbeta = np.zeros(
            (self.num_topics, self.vocab_len), dtype=np.float64
        )

        # LdaPost for the new document
        doc_max_len = len(doc_bow) if doc_bow else 0
        ldapost_new_doc = LdaPost(
            doc=doc_bow,
            lda=lda_inf_model,
            max_doc_len=doc_max_len,
            num_topics=self.num_topics,
        )

        # Average topic distribution over time slices, or use a specific one?
        # Original __getitem__ iterates all time slices and averages likelihoods/gammas.
        # This seems complex for a single doc inference without a time context.
        # A more common approach for DTM with a new doc is to infer against topics of a *specific* time slice.
        # Let's infer against the topics of the *last* time slice as a default.
        target_time_slice = self.num_time_slices - 1
        if target_time_slice < 0:
            logger.error("Model has no time slices to infer against.")
            return np.full(self.num_topics, 1.0 / self.num_topics, dtype=np.float64)

        lda_inf_model = self.make_lda_seq_slice_jax(lda_inf_model, target_time_slice)

        _ = ldapost_new_doc.fit_lda_post(
            doc_number=0,  # Only one doc
            time=target_time_slice,
            ldaseq=None,  # Not in full EM context
            lda_inference_max_iter=lda_inference_max_iter,
        )

        gamma_new_doc = ldapost_new_doc.gamma
        doc_topic_dist = gamma_new_doc / np.sum(gamma_new_doc)
        return doc_topic_dist
