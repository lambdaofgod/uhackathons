"""JAX implementation of the Numba-accelerated functions from ldaseqmodel.py"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional
import jax.scipy.optimize

logger = logging.getLogger(__name__)

# Constants
INIT_VARIANCE_CONST = 1000.0
INIT_MULT = 1000.0

# Forward pass for compute_post_mean
def _forward_pass(carry, t_idx, obs_word, fwd_variance, chain_variance, obs_variance):
    """Forward pass for computing post mean using JAX scan"""
    prev_fwd_mean = carry
    denominator = fwd_variance[t_idx-1] + chain_variance + obs_variance
    c = jnp.where(denominator != 0.0, obs_variance / denominator, 0.0)
    new_fwd_mean = c * prev_fwd_mean + (1.0 - c) * obs_word[t_idx-1]
    return new_fwd_mean, new_fwd_mean

# Backward pass for compute_post_mean
def _backward_pass(carry, t_idx, fwd_mean, fwd_variance, chain_variance):
    """Backward pass for computing post mean using JAX scan"""
    next_mean = carry
    t = t_idx  # t goes from T-1 down to 0
    fwd_var_t_plus_chain_var = fwd_variance[t] + chain_variance
    c = jnp.where(
        (chain_variance != 0.0) & (fwd_var_t_plus_chain_var != 0.0),
        chain_variance / fwd_var_t_plus_chain_var,
        0.0
    )
    new_mean = c * fwd_mean[t] + (1.0 - c) * next_mean
    return new_mean, new_mean

def _jax_compute_post_mean_scan(
    obs_word, fwd_variance_word, chain_variance, obs_variance, num_time_slices
):
    """JAX implementation of compute_post_mean using scan for better performance.
    
    Uses JAX's scan operation for more efficient forward and backward passes.
    """
    T = num_time_slices
    
    # Initialize arrays
    fwd_mean = jnp.zeros(T + 1)
    
    # Forward pass using scan
    _, fwd_mean_updates = jax.lax.scan(
        lambda carry, x: _forward_pass(carry, x, obs_word, fwd_variance_word, chain_variance, obs_variance),
        init=0.0,  # Initial value at t=0
        xs=jnp.arange(1, T+1)  # Time indices from 1 to T
    )
    
    # Update fwd_mean with scan results
    fwd_mean = fwd_mean.at[1:].set(fwd_mean_updates)
    
    # Backward pass using scan
    mean = jnp.zeros(T + 1)
    mean = mean.at[T].set(fwd_mean[T])  # Initialize last value
    
    # Scan from T-1 down to 0
    _, mean_updates = jax.lax.scan(
        lambda carry, x: _backward_pass(carry, x, fwd_mean, fwd_variance_word, chain_variance),
        init=fwd_mean[T],  # Initial value at t=T
        xs=jnp.arange(T-1, -1, -1)  # Time indices from T-1 down to 0
    )
    
    # Update mean with scan results (in reverse order)
    mean = mean.at[:T].set(mean_updates[::-1])
    
    return mean, fwd_mean

# JIT compile the function
jax_compute_post_mean_scan = jax.jit(
    _jax_compute_post_mean_scan, static_argnums=(4,)
)

# Non-JIT version for use in contexts where tracing might cause issues
def jax_compute_post_mean_scan_unjitted(
    obs_word, fwd_variance_word, chain_variance, obs_variance, num_time_slices
):
    """Unjitted version of post mean computation for use in contexts where JIT tracing causes issues."""
    return _jax_compute_post_mean_scan(
        jnp.array(obs_word), jnp.array(fwd_variance_word), 
        chain_variance, obs_variance, num_time_slices
    )

# Forward pass for mean derivative computation
def _deriv_forward_pass(carry, t_data, fwd_variance, chain_variance, obs_variance, time_obs_idx):
    """Forward pass for computing mean derivative using JAX scan"""
    t_idx, prev_deriv = t_data[0], carry
    denominator = fwd_variance[t_idx-1] + chain_variance + obs_variance
    w_factor = jnp.where(
        (obs_variance > 0.0) & (denominator != 0.0),
        obs_variance / denominator,
        0.0
    )
    
    val = w_factor * prev_deriv
    # Add 1-w_factor when t_idx-1 == time_obs_idx
    val = val + jnp.where(t_idx-1 == time_obs_idx, 1.0 - w_factor, 0.0)
    
    return val, val

# Backward pass for mean derivative computation
def _deriv_backward_pass(carry, t_data, fwd_variance, chain_variance):
    """Backward pass for computing mean derivative using JAX scan"""
    t_idx, next_deriv, curr_deriv = t_data[0], carry, t_data[1]
    fwd_var_s_plus_chain_var = fwd_variance[t_idx] + chain_variance
    w_factor_backward = jnp.where(
        (chain_variance != 0.0) & (fwd_var_s_plus_chain_var != 0.0),
        chain_variance / fwd_var_s_plus_chain_var,
        0.0
    )
    
    new_deriv = w_factor_backward * curr_deriv + (1.0 - w_factor_backward) * next_deriv
    return new_deriv, new_deriv

def jax_compute_mean_deriv(
    fwd_variance_word, time_obs_idx, chain_variance, obs_variance, num_time_slices
):
    """JAX implementation of compute_mean_deriv using scan operations.
    
    Computes d E[beta_{s,w}] / d obs_{t,w} for s = 0..T.
    """
    T = num_time_slices
    
    # Initialize derivative array
    deriv = jnp.zeros(T + 1)
    
    # Forward pass using scan
    _, deriv_updates = jax.lax.scan(
        lambda carry, x: _deriv_forward_pass(
            carry, x, fwd_variance_word, chain_variance, obs_variance, time_obs_idx
        ),
        init=0.0,  # Initial value at t=0
        xs=jnp.stack([jnp.arange(1, T+1), jnp.zeros(T)])  # Time indices from 1 to T
    )
    
    # Update deriv with scan results
    deriv = deriv.at[1:].set(deriv_updates)
    
    # Prepare data for backward pass: (t_idx, current_deriv)
    backward_data = jnp.stack([
        jnp.arange(T-1, -1, -1),  # Time indices from T-1 down to 0
        deriv[:T][::-1]  # Current deriv values in reverse order
    ])
    
    # Backward pass using scan
    _, deriv_updates = jax.lax.scan(
        lambda carry, x: _deriv_backward_pass(carry, x, fwd_variance_word, chain_variance),
        init=deriv[T],  # Initial value at t=T
        xs=backward_data
    )
    
    # Update deriv with scan results (in reverse order)
    deriv = deriv.at[:T].set(deriv_updates[::-1])
    
    return deriv

# Vectorized computation of all mean derivatives at once
def jax_compute_all_mean_derivs(
    fwd_variance_word, chain_variance, obs_variance, num_time_slices
):
    """Compute all mean derivatives for all observation times at once.
    
    Returns a matrix of shape (num_time_slices, num_time_slices + 1).
    """
    T = num_time_slices
    
    # Vectorize the computation over time_obs_idx
    compute_deriv_for_time = lambda time_idx: jax_compute_mean_deriv(
        fwd_variance_word, time_idx, chain_variance, obs_variance, T
    )
    
    # Use vmap to compute derivatives for all time indices at once
    all_derivs = jax.vmap(compute_deriv_for_time)(jnp.arange(T))
    
    return all_derivs

# Vectorized computation for objective function terms
def _compute_objective_terms(
    t_idx, mean, variance, word_counts, totals, zeta, chain_variance
):
    """Compute objective function terms for a single time index."""
    m_wt_plus_1 = mean[t_idx + 1]
    m_wt = mean[t_idx]
    
    # Word count term
    term1 = word_counts[t_idx] * m_wt_plus_1
    
    # Zeta term
    exp_term = jnp.exp(m_wt_plus_1 + variance[t_idx + 1] / 2.0)
    zeta_term = jnp.where(
        zeta[t_idx] > 0.0,
        -totals[t_idx] * exp_term / zeta[t_idx],
        0.0
    )
    
    # Chain variance term
    chain_term = jnp.where(
        chain_variance > 0.0,
        -(jnp.power(m_wt_plus_1 - m_wt, 2) / (2 * chain_variance)) 
        + (variance[t_idx + 1] / chain_variance) 
        + jnp.log(chain_variance),
        0.0
    )
    
    return term1 + zeta_term + chain_term

@jax.jit
def jax_f_obs(
    x_obs_w, word_counts_w, totals_time, variance_word_fixed, fwd_variance_word_fixed,
    chain_variance, obs_variance_scalar, num_time_slices, zeta_topic_fixed
):
    """JAX implementation of f_obs with vectorized operations.
    
    Objective function for optimizing obs values.
    """
    T = num_time_slices
    
    # Compute mean based on current x_obs_w
    mean, _ = jax_compute_post_mean_scan_unjitted(
        x_obs_w, fwd_variance_word_fixed, chain_variance, obs_variance_scalar, T
    )
    
    # Vectorize objective term computation over time indices
    objective_terms = jax.vmap(
        lambda t: _compute_objective_terms(
            t, mean, variance_word_fixed, word_counts_w, 
            totals_time, zeta_topic_fixed, chain_variance
        )
    )(jnp.arange(T))
    
    # Sum all terms
    current_objective_val = jnp.sum(objective_terms)
    
    # Prior term for m_w[0]
    prior_term = jnp.where(
        (chain_variance > 0.0) & (INIT_MULT > 0.0),
        -jnp.power(mean[0], 2) / (2 * INIT_MULT * chain_variance),
        0.0
    )
    
    current_objective_val += prior_term
    
    return -current_objective_val  # Negative for minimization

# Vectorized computation for gradient terms
def _compute_gradient_part1(
    u_time_idx, word_counts, totals, temp_vect, zeta, mean_deriv, T
):
    """Compute part 1 of the gradient for a single time index."""
    dmean_u_plus_1_dobs_t = mean_deriv[u_time_idx + 1]
    
    # Word count term
    count_term = word_counts[u_time_idx] * dmean_u_plus_1_dobs_t
    
    # Zeta term
    exp_term_w_u = temp_vect[u_time_idx]
    zeta_u = zeta[u_time_idx]
    
    zeta_term = jnp.where(
        zeta_u > 0.0,
        -totals[u_time_idx] * exp_term_w_u * dmean_u_plus_1_dobs_t * 
        (zeta_u - exp_term_w_u) / (zeta_u * zeta_u),
        0.0
    )
    
    return count_term + zeta_term

def _compute_gradient_part2(
    u_time_idx, mean, mean_deriv, chain_variance
):
    """Compute part 2 of the gradient for a single time index."""
    mean_u_plus_1 = mean[u_time_idx + 1]
    mean_u = mean[u_time_idx]
    dmean_u_plus_1_dobs_t = mean_deriv[u_time_idx + 1]
    dmean_u_dobs_t = mean_deriv[u_time_idx]
    
    return jnp.where(
        chain_variance > 0.0,
        -((mean_u_plus_1 - mean_u) / chain_variance) * (dmean_u_plus_1_dobs_t - dmean_u_dobs_t),
        0.0
    )

@jax.jit
def jax_df_obs(
    x_obs_w, word_counts_w, totals_time, variance_word_fixed, fwd_variance_word_fixed,
    chain_variance, obs_variance_scalar, num_time_slices, zeta_topic_fixed
):
    """JAX implementation of df_obs with vectorized operations.
    
    Gradient function for optimizing obs values.
    """
    T = num_time_slices
    
    # Compute mean based on current x_obs_w
    mean, _ = jax_compute_post_mean_scan_unjitted(
        x_obs_w, fwd_variance_word_fixed, chain_variance, obs_variance_scalar, T
    )
    
    # Compute all mean derivatives at once
    mean_deriv_mtx = jax_compute_all_mean_derivs(
        fwd_variance_word_fixed, chain_variance, obs_variance_scalar, T
    )
    
    # Precompute temp_vect: exp(mean[u+1] + var[u+1]/2) for u=0..T-1
    temp_vect = jnp.exp(mean[1:T+1] + variance_word_fixed[1:T+1] / 2.0)
    
    # Initialize gradient array
    grad = jnp.zeros(T)
    
    # For each observation time t
    for t_obs_idx in range(T):
        mean_deriv = mean_deriv_mtx[t_obs_idx]
        
        # Part 1: Word count and zeta terms (vectorized)
        part1_terms = jax.vmap(
            lambda u: _compute_gradient_part1(
                u, word_counts_w, totals_time, temp_vect, 
                zeta_topic_fixed, mean_deriv, T
            )
        )(jnp.arange(T))
        
        # Part 2: Chain variance terms (vectorized)
        part2_terms = jax.vmap(
            lambda u: _compute_gradient_part2(
                u, mean, mean_deriv, chain_variance
            )
        )(jnp.arange(T))
        
        # Part 3: Prior term for m_w[0]
        prior_term = jnp.where(
            (chain_variance > 0.0) & (INIT_MULT > 0.0),
            -(mean[0] / (INIT_MULT * chain_variance)) * mean_deriv[0],
            0.0
        )
        
        # Sum all terms
        deriv_val = jnp.sum(part1_terms) + jnp.sum(part2_terms) + prior_term
        grad = grad.at[t_obs_idx].set(deriv_val)
    
    return -grad  # Negative for minimization

# JIT-compiled optimization function
@jax.jit
def _optimize_obs_word(
    initial_obs_w, word_counts, totals, variance, fwd_variance, 
    chain_variance, obs_variance, num_time_slices, zeta
):
    """JIT-compiled function to optimize obs values for a single word."""
    
    # Define the objective function and its gradient for JAX optimizer
    def objective_and_grad(x):
        # Compute objective value
        value = jax_f_obs(
            x, word_counts, totals, variance, fwd_variance,
            chain_variance, obs_variance, num_time_slices, zeta
        )
        # Compute gradient
        grad = jax_df_obs(
            x, word_counts, totals, variance, fwd_variance,
            chain_variance, obs_variance, num_time_slices, zeta
        )
        return value, grad
    
    # Use JAX's BFGS optimizer (similar to CG but often more efficient)
    result = jax.scipy.optimize.minimize(
        fun=lambda x: objective_and_grad(x)[0],
        x0=initial_obs_w,
        jac=lambda x: objective_and_grad(x)[1],
        method='BFGS',
        options={'gtol': 1e-3}
    )
    
    return result.x

def update_obs_jax(
    sslm, sstats: np.ndarray, totals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JAX-accelerated version of update_obs that optimizes the bound with respect to 
    the observed variables (sslm.obs).
    
    This implementation uses JAX for faster computation and vectorization where possible.
    
    Parameters
    ----------
    sslm : object
        The State Space Language Model object containing the model state
    sstats : numpy.ndarray
        Sufficient statistics for a particular topic
    totals : numpy.ndarray
        The totals for each time slice
        
    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The updated obs and zeta values
    """
    if sslm.vocab_len is None or sslm.num_time_slices is None:
        logging.error(
            "sslm not properly initialized: vocab_len or num_time_slices is None"
        )
        return sslm.obs, sslm.zeta
        
    W = sslm.vocab_len
    T = sslm.num_time_slices
    
    # Constants for optimization
    OBS_NORM_CUTOFF = 2.0
    
    # Convert totals to JAX array once (used for all words)
    jax_totals = jnp.array(totals)
    
    norm_cutoff_obs_cache: Optional[np.ndarray] = None
    
    # Process words one at a time (future improvement: batch processing)
    for w_idx in range(W):
        word_counts_w = sstats[w_idx, :]
        counts_norm = np.linalg.norm(word_counts_w)
        
        current_obs_w_optimized: np.ndarray
        run_optimization = True
        
        if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs_cache is not None:
            current_obs_w_optimized = np.copy(norm_cutoff_obs_cache)
            run_optimization = False
            
        if run_optimization:
            word_counts_w_for_opt = word_counts_w
            if counts_norm < OBS_NORM_CUTOFF:
                word_counts_w_for_opt = np.zeros_like(word_counts_w)
                
            # Fixed data for the current word - convert to JAX arrays once
            jax_word_counts = jnp.array(word_counts_w_for_opt)
            jax_variance = jnp.array(sslm.variance[w_idx, :])
            jax_fwd_variance = jnp.array(sslm.fwd_variance[w_idx, :])
            jax_initial_obs = jnp.array(sslm.obs[w_idx, :])
            jax_zeta = jnp.array(sslm.zeta)
            
            try:
                # Use the JIT-compiled optimization function
                optimized_result = _optimize_obs_word(
                    jax_initial_obs, 
                    jax_word_counts, 
                    jax_totals,
                    jax_variance, 
                    jax_fwd_variance,
                    sslm.chain_variance, 
                    sslm.obs_variance,
                    T, 
                    jax_zeta
                )
                
                # Convert result back to numpy
                current_obs_w_optimized = np.array(optimized_result)
                
            except Exception as e:
                logging.error(f"JAX optimization failed for word {w_idx}: {e}")
                current_obs_w_optimized = sslm.obs[w_idx, :]
                
            if counts_norm < OBS_NORM_CUTOFF:
                norm_cutoff_obs_cache = np.copy(current_obs_w_optimized)
                
        sslm.obs[w_idx, :] = current_obs_w_optimized
    
    # Update mean and fwd_mean for all words
    for w_idx in range(W):
        # Convert to JAX arrays
        jax_obs = jnp.array(sslm.obs[w_idx, :])
        jax_fwd_variance = jnp.array(sslm.fwd_variance[w_idx, :])
        
        # Compute means
        mean, fwd_mean = jax_compute_post_mean_scan_unjitted(
            jax_obs,
            jax_fwd_variance,
            sslm.chain_variance,
            sslm.obs_variance,
            T
        )
        
        # Copy results back to numpy arrays
        sslm.mean[w_idx, :] = np.array(mean)
        sslm.fwd_mean[w_idx, :] = np.array(fwd_mean)
        
    # Update zeta
    sslm.zeta = sslm.update_zeta()
    
    return sslm.obs, sslm.zeta
