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
"""

import logging
from typing import Tuple, List, Any, Optional

import numpy as np # Still used for initial setup and some non-JAX parts
import jax
import jax.numpy as jnp
import optax
from scipy.special import gammaln # For LdaPost, if used, or bound calculations

from gensim import utils, matutils
from gensim.models import ldamodel # May still be used for E-step components

logger = logging.getLogger(__name__)

# JAX-specific global constants (if any, prefer passing as args or class members)
INIT_VARIANCE_CONST_JAX = 1000.0
INIT_MULT_JAX = 1000.0
OBS_NORM_CUTOFF_JAX = 2.0 # Default, can be instance variable
DEFAULT_JAX_OBS_OPTIMIZER_LR = 0.01
DEFAULT_JAX_OBS_OPTIMIZER_STEPS = 50


@jax.jit
def jax_compute_post_mean_scan(
    obs_word: jnp.ndarray,  # (T)
    fwd_variance_word: jnp.ndarray,  # (T+1)
    chain_variance: float,
    obs_variance: float,
    num_time_slices: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]: # mean_word_out (T+1), fwd_mean_word_out (T+1)
    T = num_time_slices
    dtype = obs_word.dtype

    # --- Forward pass for fwd_mean using scan ---
    def fwd_scan_body(carry_fwd_mean_prev, scan_inputs_t_minus_1):
        obs_val_t_minus_1, fwd_var_val_t_minus_1 = scan_inputs_t_minus_1
        
        denominator = fwd_var_val_t_minus_1 + chain_variance + obs_variance
        # Add epsilon to denominator to prevent 0/0 -> NaN if all terms are 0
        c_factor = jnp.where(denominator != 0.0, obs_variance / (denominator + 1e-9), 0.0)
        
        fwd_mean_curr = (
            c_factor * carry_fwd_mean_prev
            + (1.0 - c_factor) * obs_val_t_minus_1
        )
        return fwd_mean_curr, fwd_mean_curr

    initial_fwd_mean_carry = 0.0 
    scan_fwd_inputs = (obs_word, fwd_variance_word[:-1]) 
    
    _, fwd_mean_values_1_to_T = jax.lax.scan(fwd_scan_body, initial_fwd_mean_carry, scan_fwd_inputs)
    
    fwd_mean_word_out = jnp.zeros(T + 1, dtype=dtype).at[1:].set(fwd_mean_values_1_to_T)

    # --- Backward pass for smooth_mean using scan ---
    def bwd_scan_body(carry_mean_next, scan_inputs_t):
        fwd_mean_val_t, fwd_var_val_t = scan_inputs_t
        
        fwd_var_t_plus_chain_var = fwd_var_val_t + chain_variance
        # Add epsilon to denominator
        c_factor_backward = jnp.where(
            (chain_variance != 0.0) & (fwd_var_t_plus_chain_var != 0.0),
            chain_variance / (fwd_var_t_plus_chain_var + 1e-9),
            0.0
        )
        
        mean_curr = (
            c_factor_backward * fwd_mean_val_t
            + (1.0 - c_factor_backward) * carry_mean_next
        )
        return mean_curr, mean_curr

    initial_bwd_mean_carry = fwd_mean_word_out[T]
    scan_bwd_inputs = (jnp.flip(fwd_mean_word_out[:-1]), jnp.flip(fwd_variance_word[:-1]))

    _, mean_values_T_minus_1_to_0_rev = jax.lax.scan(bwd_scan_body, initial_bwd_mean_carry, scan_bwd_inputs)
    
    mean_word_out = jnp.zeros(T + 1, dtype=dtype)
    mean_word_out = mean_word_out.at[T].set(fwd_mean_word_out[T])
    mean_word_out = mean_word_out.at[:T].set(jnp.flip(mean_values_T_minus_1_to_0_rev))
    
    return mean_word_out, fwd_mean_word_out


@jax.jit
def jax_compute_post_variance_scan(
    _placeholder_for_vmap: jnp.ndarray, # Used to vmap W times if needed
    obs_variance_scalar: float,
    chain_variance_scalar: float,
    num_time_slices: int,
    dtype: jnp.dtype = jnp.float64
) -> Tuple[jnp.ndarray, jnp.ndarray]: # variance_out (T+1), fwd_variance_out (T+1)
    T = num_time_slices

    # --- Forward pass for fwd_variance ---
    def fwd_var_scan_body(carry_fwd_var_prev, _): # Loop T times
        denominator = carry_fwd_var_prev + chain_variance_scalar + obs_variance_scalar
        c_factor = jnp.where(denominator != 0.0, obs_variance_scalar / (denominator + 1e-9), 0.0)
        fwd_var_curr = c_factor * (carry_fwd_var_prev + chain_variance_scalar)
        return fwd_var_curr, fwd_var_curr

    initial_fwd_var_carry = chain_variance_scalar * INIT_VARIANCE_CONST_JAX
    _, fwd_var_values_1_to_T = jax.lax.scan(fwd_var_scan_body, initial_fwd_var_carry, jnp.arange(T))
    
    fwd_variance_out = jnp.zeros(T + 1, dtype=dtype)
    fwd_variance_out = fwd_variance_out.at[0].set(initial_fwd_var_carry)
    fwd_variance_out = fwd_variance_out.at[1:].set(fwd_var_values_1_to_T)

    # --- Backward pass for variance ---
    def bwd_var_scan_body(carry_var_next, fwd_var_val_t):
        c_bwd_factor_den = fwd_var_val_t + chain_variance_scalar
        c_bwd_ratio = jnp.where(c_bwd_factor_den != 0.0, fwd_var_val_t / (c_bwd_factor_den + 1e-9), 0.0)
        c_bwd_sq = jnp.power(c_bwd_ratio, 2)
        c_bwd_sq = jnp.where(fwd_var_val_t > 1e-9, c_bwd_sq, 0.0) # if fwd_var_val_t is ~0, c_bwd_sq is 0

        var_curr = (c_bwd_sq * (carry_var_next - chain_variance_scalar)) + (
            (1.0 - c_bwd_sq) * fwd_var_val_t
        )
        return var_curr, var_curr

    initial_bwd_var_carry = fwd_variance_out[T]
    scan_bwd_inputs_var = jnp.flip(fwd_variance_out[:-1])
    _, var_values_T_minus_1_to_0_rev = jax.lax.scan(bwd_var_scan_body, initial_bwd_var_carry, scan_bwd_inputs_var)
    
    variance_out = jnp.zeros(T + 1, dtype=dtype)
    variance_out = variance_out.at[T].set(fwd_variance_out[T])
    variance_out = variance_out.at[:T].set(jnp.flip(var_values_T_minus_1_to_0_rev))
    
    return variance_out, fwd_variance_out


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
        x_obs_w, fwd_variance_word_fixed, chain_variance,
        obs_variance_scalar, num_time_slices
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
    
    current_objective_val = jax.lax.fori_loop(0, T, objective_loop_body, current_objective_val)

    # Prior term for m_w[0]
    safe_chain_variance_prior = chain_variance + 1e-9
    safe_init_mult_jax = INIT_MULT_JAX + 1e-9 # Ensure INIT_MULT_JAX is positive
    current_objective_val -= jnp.power(mean_w[0], 2) / (
        2 * safe_init_mult_jax * safe_chain_variance_prior
    )
    
    # Optimizers minimize, original f_obs returned -(objective for maximization)
    # So, if jax_objective_for_word_obs calculates the positive objective, return -objective
    return -current_objective_val 

# Pre-compile the gradient function
# Note: JITting jax.grad(jax_objective_for_word_obs) directly.
# argnums=0 means differentiate w.r.t. x_obs_w
grad_fn_jax_objective_for_word_obs = jax.jit(jax.grad(jax_objective_for_word_obs, argnums=0))


class sslm_jax(utils.SaveLoad):
    """
    JAX-based State Space Language Model for DTM.
    Corresponds to the `sslm` class in the original LdaSeqModel.
    """
    def __init__(
        self,
        vocab_len: int,
        num_time_slices: int,
        num_topics: int, # For context, not directly used in all methods here
        obs_variance: float = 0.5,
        chain_variance: float = 0.005,
        obs_optimizer_lr: float = DEFAULT_JAX_OBS_OPTIMIZER_LR,
        obs_optimizer_steps: int = DEFAULT_JAX_OBS_OPTIMIZER_STEPS,
        dtype: jnp.dtype = jnp.float64
    ):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.num_topics = num_topics # Unused in this snippet but part of original
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.dtype = dtype

        self.obs_optimizer_lr = obs_optimizer_lr
        self.obs_optimizer_steps = obs_optimizer_steps
        self.obs_norm_cutoff = OBS_NORM_CUTOFF_JAX # Can be configured

        # Initialize JAX arrays (or convert from numpy if needed)
        self.obs = jnp.zeros((vocab_len, num_time_slices), dtype=dtype)
        self.e_log_prob = jnp.zeros((vocab_len, num_time_slices), dtype=dtype)
        self.mean = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.fwd_mean = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.variance = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.fwd_variance = jnp.zeros((vocab_len, num_time_slices + 1), dtype=dtype)
        self.zeta = jnp.zeros(num_time_slices, dtype=dtype)
        
        # Other DIM-related attributes from original sslm are omitted for now
        # self.w_phi_l, self.w_phi_sum, etc.

    def update_zeta_jax(self) -> jnp.ndarray:
        # self.mean is (W, T+1), self.variance is (W, T+1)
        # self.zeta is (T)
        exp_terms = jnp.exp(self.mean[:, 1:] + self.variance[:, 1:] / 2.0) # Shape (W, T)
        new_zeta = jnp.sum(exp_terms, axis=0) # Sum over W, result shape (T)
        return new_zeta

    def compute_expected_log_prob_jax(self) -> jnp.ndarray:
        # self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        # Vectorized:
        # self.mean[:, 1:] is (W, T)
        # jnp.log(self.zeta) is (T), needs broadcasting for subtraction with (W,T)
        log_zeta_broadcast = jnp.log(self.zeta[jnp.newaxis, :] + 1e-9) # (1,T)
        new_e_log_prob = self.mean[:, 1:] - log_zeta_broadcast
        return new_e_log_prob

    def sslm_counts_init_jax(self, sstats_topic: np.ndarray): # sstats_topic is for one topic (Vocab,)
        W = self.vocab_len
        T = self.num_time_slices

        log_norm_counts = np.copy(sstats_topic) # numpy operations for init
        log_norm_counts /= np.sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= np.sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)

        self.obs = jnp.array(np.tile(log_norm_counts[:, np.newaxis], (1, T)), dtype=self.dtype)

        # Compute post variance (vmapped)
        # Placeholder for vmap: jnp.arange(W) - this argument is not used by the function itself
        # but tells vmap to run W times.
        vmap_compute_var = jax.vmap(
            jax_compute_post_variance_scan, in_axes=(None, None, None, None, None), out_axes=0
        )
        all_vars, all_fwd_vars = vmap_compute_var(
            jnp.arange(W), self.obs_variance, self.chain_variance, T, self.dtype
        )
        self.variance = all_vars
        self.fwd_variance = all_fwd_vars
        
        # Compute post mean (vmapped)
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


    def update_obs_jax(self, sstats_topic_all_times: jnp.ndarray, totals_all_times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        W = self.vocab_len
        T = self.num_time_slices
        optimizer = optax.adam(learning_rate=self.obs_optimizer_lr)
        
        new_obs_list = []
        norm_cutoff_obs_cache = None # Reset per call

        for w_idx in range(W): # Python loop over words
            word_counts_w = sstats_topic_all_times[w_idx, :] # (T,)
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
                # Static args for the current word's optimization
                static_args_for_opt_tuple = (
                    word_counts_w_for_opt,
                    totals_all_times,
                    self.variance[w_idx, :],
                    self.fwd_variance[w_idx, :],
                    self.chain_variance,
                    self.obs_variance,
                    T,
                    self.zeta,
                )

                # JIT-compiled optimization loop for a single obs_w vector
                # grad_fn_jax_objective_for_word_obs is already JITted
                @jax.jit
                def optimize_single_obs_w_loop(initial_obs_w, *args_tuple_for_loss):
                    opt_state = optimizer.init(initial_obs_w)
                    
                    def opt_step(_, state_tuple): # Loop index not used
                        current_obs, current_opt_state = state_tuple
                        grads = grad_fn_jax_objective_for_word_obs(current_obs, *args_tuple_for_loss)
                        updates, new_opt_state = optimizer.apply_updates(grads, current_opt_state, current_obs)
                        new_obs = optax.apply_updates(current_obs, updates)
                        return new_obs, new_opt_state

                    final_obs_w, _ = jax.lax.fori_loop(
                        0, self.obs_optimizer_steps, opt_step, (initial_obs_w, opt_state)
                    )
                    return final_obs_w
                
                current_obs_w_optimized = optimize_single_obs_w_loop(initial_obs_w_val, *static_args_for_opt_tuple)

                if counts_norm < self.obs_norm_cutoff and run_optimization:
                     norm_cutoff_obs_cache = jnp.copy(current_obs_w_optimized)
            
            new_obs_list.append(current_obs_w_optimized)

        self.obs = jnp.stack(new_obs_list, axis=0)

        # Recompute self.mean and self.fwd_mean for all words using the new self.obs
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

    def compute_bound_jax(self, sstats_topic_all_times: jnp.ndarray, totals_all_times: jnp.ndarray) -> float:
        W = self.vocab_len
        T = self.num_time_slices
        safe_chain_variance = self.chain_variance + 1e-9

        # Initial term related to variance at boundaries
        # Original: sum(self.variance[w][0] - self.variance[w][t] for w in range(w)) / 2 * chain_variance
        # Assuming 't' in original meant T (num_time_slices)
        bound_val = jnp.sum(self.variance[:, 0] - self.variance[:, T]) / (2 * safe_chain_variance)

        # Loop over time slices t = 0 to T-1 (for sstats, totals, zeta)
        # Corresponds to mean/variance indices t+1 and t
        def bound_time_loop_body(t_loop_idx, current_bound_accum):
            # Contributions from all words for this time slice t_loop_idx
            # mean[:, t_loop_idx + 1] is m_{t+1}
            # mean[:, t_loop_idx] is m_{t}
            # variance[:, t_loop_idx + 1] is v_{t+1}
            # sstats_topic_all_times[:, t_loop_idx] is sstats_w,t
            
            m_t_plus_1 = self.mean[:, t_loop_idx + 1]
            m_t = self.mean[:, t_loop_idx]
            v_t_plus_1 = self.variance[:, t_loop_idx + 1]
            sstats_w_t = sstats_topic_all_times[:, t_loop_idx]

            # Term 1 (sum over w)
            term1_w_contrib = (jnp.power(m_t_plus_1 - m_t, 2) / (2 * safe_chain_variance)) - \
                              (v_t_plus_1 / safe_chain_variance) - \
                              jnp.log(safe_chain_variance)
            sum_term1_w = jnp.sum(term1_w_contrib)
            
            # Term 2 (sum over w)
            term2_w_contrib = sstats_w_t * m_t_plus_1
            sum_term2_w = jnp.sum(term2_w_contrib)
            
            # Entropy term (sum over w)
            # Add epsilon to v_t_plus_1 for log to prevent log(0) or log(<0)
            ent_w_contrib = jnp.log(jnp.abs(v_t_plus_1) + 1e-9) / 2.0 
            sum_ent_w = jnp.sum(ent_w_contrib)

            # Term 3 (scalar for this time slice)
            term3_scalar = -totals_all_times[t_loop_idx] * jnp.log(self.zeta[t_loop_idx] + 1e-9)
            
            current_bound_accum += sum_term2_w + term3_scalar + sum_ent_w - sum_term1_w
            return current_bound_accum

        bound_val = jax.lax.fori_loop(0, T, bound_time_loop_body, bound_val)
        return bound_val


    def fit_sslm_jax(self, sstats_topic_all_times: jnp.ndarray, totals_all_times: jnp.ndarray,
                     sslm_fit_threshold: float = 1e-6, sslm_max_iter: int = 2) -> float:
        W = self.vocab_len
        T = self.num_time_slices
        
        # Initial computation of variance (once per fit_sslm call)
        vmap_compute_var = jax.vmap(
            jax_compute_post_variance_scan, in_axes=(None, None, None, None, None), out_axes=0
        )
        all_vars, all_fwd_vars = vmap_compute_var(
            jnp.arange(W), self.obs_variance, self.chain_variance, T, self.dtype
        )
        self.variance = all_vars
        self.fwd_variance = all_fwd_vars

        # Initial bound calculation
        # Need mean and zeta consistent with current obs and new variance
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
            
            # Update observations (M-step part 1)
            self.obs, self.zeta = self.update_obs_jax(sstats_topic_all_times, totals_all_times)
            # self.mean and self.fwd_mean are updated within update_obs_jax
            
            # Recompute bound
            bound = self.compute_bound_jax(sstats_topic_all_times, totals_all_times)
            
            convergence = jnp.fabs((bound - old_bound) / (old_bound + 1e-9)) # Add epsilon to old_bound
            logger.info(
                "sslm_jax iter %i, bound %f, convergence %f",
                iter_num + 1, bound, convergence
            )
            if convergence < sslm_fit_threshold:
                break
        
        self.e_log_prob = self.compute_expected_log_prob_jax()
        return bound


class LdaSeqModelJax(utils.SaveLoad):
    """
    JAX-based LdaSeqModel (Dynamic Topic Model).
    This is a skeleton and needs further implementation, especially for the E-step
    and coordination with Gensim's LdaModel if parts of it are reused.
    """
    def __init__(
        self,
        corpus=None, # Keep similar API to original
        time_slice: Optional[List[int]] = None,
        id2word: Optional[Any] = None, # gensim.corpora.Dictionary
        num_topics: int = 10,
        alphas: float = 0.01, # Symmetric alpha
        chain_variance: float = 0.005,
        obs_variance: float = 0.5,
        # ... other LdaSeqModel parameters ...
        random_state_seed: int = 0, # For JAX PRNG key
        dtype: jnp.dtype = jnp.float64,
        # JAX specific SSLM params
        sslm_obs_optimizer_lr: float = DEFAULT_JAX_OBS_OPTIMIZER_LR,
        sslm_obs_optimizer_steps: int = DEFAULT_JAX_OBS_OPTIMIZER_STEPS,
        # E-step related params from original
        lda_inference_max_iter=25,
        em_min_iter=6,
        em_max_iter=20,
        chunksize=100,
        passes=10 # For initial LDA model if used
    ):
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError("Need corpus or id2word for vocab size.")
        
        self.vocab_len = 0
        if self.id2word:
            self.vocab_len = len(self.id2word)
        # TODO: Infer vocab_len from corpus if id2word is None (like original)

        self.num_topics = num_topics
        self.time_slice = time_slice
        self.num_time_slices = len(time_slice) if time_slice else 0
        self.alphas_np = np.full(num_topics, alphas, dtype=np.float64) # For Gensim LDA
        self.dtype = dtype
        self.key = jax.random.PRNGKey(random_state_seed)

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
                dtype=self.dtype
            )
            self.topic_chains_jax.append(sslm_instance)

        self.gammas: Optional[jnp.ndarray] = None # Variational parameters for doc-topic

        if corpus is not None and time_slice is not None:
            # Initialize with a standard LDA model (Gensim's)
            # This part is similar to original LdaSeqModel
            logger.info("Initializing LdaSeqModelJax with a standard LDA model...")
            lda_model_init = ldamodel.LdaModel(
                corpus,
                id2word=self.id2word,
                num_topics=self.num_topics,
                passes=passes, # from params
                alpha=self.alphas_np, # from params
                random_state=np.random.RandomState(random_state_seed), # Gensim LDA uses numpy random state
                dtype=np.float64, # Gensim LDA typically uses float64
            )
            # sstats are (num_terms, num_topics) in original, beta for time 0
            initial_sstats_beta = np.transpose(lda_model_init.state.sstats) # (vocab_len, num_topics)
            
            self.init_ldaseq_ss_jax(initial_sstats_beta)

            # Fit DTM using JAX SSLM
            # self.fit_lda_seq_jax(corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize)
            logger.info("LdaSeqModelJax initialized. Call fit_lda_seq_jax() to train.")


    def init_ldaseq_ss_jax(self, init_suffstats_beta: np.ndarray): # (vocab_len, num_topics)
        logger.info("Initializing JAX SSLM topic chains...")
        for k, chain_jax in enumerate(self.topic_chains_jax):
            sstats_for_topic_k = init_suffstats_beta[:, k] # (vocab_len,)
            chain_jax.sslm_counts_init_jax(sstats_for_topic_k)
        logger.info("JAX SSLM topic chains initialized.")

    def fit_lda_seq_topics_jax(self, topic_suffstats_all_topics: List[jnp.ndarray]) -> float:
        """ M-step: Fit the SSLM for each topic. """
        total_topic_bound = 0.0
        for k, chain_jax in enumerate(self.topic_chains_jax):
            logger.info(f"Fitting JAX SSLM for topic {k}...")
            # sstats for topic k, all time slices: (vocab_len, num_time_slices)
            sstats_k = topic_suffstats_all_topics[k] 
            # totals for topic k, all time slices: (num_time_slices,)
            # In original, totals were sum over vocab of sstats for a given time slice.
            # Assuming sstats_k is (vocab_len, num_time_slices)
            totals_k = jnp.sum(sstats_k, axis=0)

            lhood_term = chain_jax.fit_sslm_jax(sstats_k, totals_k)
            total_topic_bound += lhood_term
        return total_topic_bound

    # Placeholder for the E-step. This is complex as it interacts with document processing
    # and potentially Gensim's LdaModel components for variational inference per document.
    # def lda_seq_infer_jax(self, corpus, ...):
    #     """ E-step: Perform inference for document variational parameters (gammas)
    #         and collect sufficient statistics for topics.
    #     """
    #     # ... This would involve iterating through documents, time slices,
    #     #     performing LDA-like inference for each document given current topic_chains_jax,
    #     #     and accumulating sufficient statistics (topic_suffstats).
    #     #     The original LdaPost class logic would need to be adapted or replaced.
    #     bound = 0.0
    #     # topic_suffstats = [jnp.zeros((self.vocab_len, self.num_time_slices), dtype=self.dtype) for _ in range(self.num_topics)]
    #     # gammas = jnp.zeros((num_documents, self.num_topics), dtype=self.dtype)
    #     # return bound, gammas, topic_suffstats
    #     raise NotImplementedError("E-step (lda_seq_infer_jax) is not fully implemented.")

    # def fit_lda_seq_jax(self, corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize):
    #     """ Main EM training loop for LdaSeqModelJax. """
    #     logger.info("Starting LdaSeqModelJax EM training...")
    #     # ... EM loop structure similar to original ...
    #     # In each iteration:
    #     # 1. E-step: Call lda_seq_infer_jax -> get bound_e_step, gammas, topic_suffstats
    #     # 2. M-step: Call fit_lda_seq_topics_jax(topic_suffstats) -> get bound_m_step
    #     # 3. Update total bound, check convergence.
    #     raise NotImplementedError("fit_lda_seq_jax is not fully implemented.")

    def print_topic_times_jax(self, topic_idx: int, top_terms: int = 10) -> List[List[Tuple[str, float]]]:
        """ Print topic evolution over time for a given topic_idx. """
        if not self.id2word:
            logger.warning("id2word not available, cannot print topics with words.")
            return []
        
        chain = self.topic_chains_jax[topic_idx]
        # chain.e_log_prob is (vocab_len, num_time_slices)
        # Convert to numpy for easier processing with id2word
        e_log_prob_np = np.array(chain.e_log_prob) # JAX array to NumPy
        
        topics_over_time = []
        for t in range(self.num_time_slices):
            log_probs_t = e_log_prob_np[:, t]
            # Normalize to get probabilities (exp and sum)
            probs_t = np.exp(log_probs_t)
            probs_t /= np.sum(probs_t)
            
            best_n_indices = np.argsort(probs_t)[::-1][:top_terms]
            topic_terms = [(self.id2word[idx], float(probs_t[idx])) for idx in best_n_indices]
            topics_over_time.append(topic_terms)
        return topics_over_time

    # Other methods like __getitem__, doc_topics, etc., would need adaptation.

# Example of how LdaPost might be used or adapted if E-step is complex
# class LdaPostJax(utils.SaveLoad):
#    ... (This would be a JAX/Numpy hybrid if interacting with Gensim LDA for E-step)

