"""JAX implementation of the Numba-accelerated functions from ldaseqmodel.py"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
from scipy import optimize

logger = logging.getLogger(__name__)

# Constants
INIT_VARIANCE_CONST = 1000.0
INIT_MULT = 1000.0

def _jax_compute_post_mean_scan_uncompiled(
    obs_word, fwd_variance_word, chain_variance, obs_variance, num_time_slices
):
    """JAX implementation of compute_post_mean using scan for better performance.
    
    Parameters
    ----------
    obs_word : jnp.ndarray
        Observations for a single word across all time slices, shape (num_time_slices,)
    fwd_variance_word : jnp.ndarray
        Forward variances for a single word, shape (num_time_slices + 1,)
    chain_variance : float
        Variance parameter for the state transition
    obs_variance : float
        Variance parameter for the observations
    num_time_slices : int
        Number of time slices
        
    Returns
    -------
    (jnp.ndarray, jnp.ndarray)
        Mean and forward mean arrays, both of shape (num_time_slices + 1,)
    """
    T = num_time_slices
    
    # Initialize arrays
    fwd_mean = jnp.zeros(T + 1)
    
    # Forward pass
    fwd_mean = fwd_mean.at[0].set(0.0)
    for t in range(1, T + 1):
        denominator = fwd_variance_word[t - 1] + chain_variance + obs_variance
        c = jnp.where(denominator != 0.0, obs_variance / denominator, 0.0)
        fwd_mean = fwd_mean.at[t].set(c * fwd_mean[t - 1] + (1.0 - c) * obs_word[t - 1])
    
    # Backward pass
    mean = jnp.zeros(T + 1)
    mean = mean.at[T].set(fwd_mean[T])
    
    for t in range(T - 1, -1, -1):
        fwd_var_t_plus_chain_var = fwd_variance_word[t] + chain_variance
        c = jnp.where(
            (chain_variance != 0.0) & (fwd_var_t_plus_chain_var != 0.0),
            chain_variance / fwd_var_t_plus_chain_var,
            0.0
        )
        mean = mean.at[t].set(c * fwd_mean[t] + (1.0 - c) * mean[t + 1])
    
    return mean, fwd_mean

# JIT compile the function
jax_compute_post_mean_scan = jax.jit(
    _jax_compute_post_mean_scan_uncompiled, static_argnums=(4,)
)

# Non-JIT version for use in contexts where tracing might cause issues
def jax_compute_post_mean_scan_unjitted(
    obs_word, fwd_variance_word, chain_variance, obs_variance, num_time_slices
):
    """Unjitted version of post mean computation for use in contexts where JIT tracing causes issues."""
    return _jax_compute_post_mean_scan_uncompiled(
        jnp.array(obs_word), jnp.array(fwd_variance_word), 
        chain_variance, obs_variance, num_time_slices
    )

def jax_compute_mean_deriv(
    fwd_variance_word, time_obs_idx, chain_variance, obs_variance, num_time_slices
):
    """JAX implementation of compute_mean_deriv.
    
    Computes d E[beta_{s,w}] / d obs_{t,w} for s = 0..T.
    
    Parameters
    ----------
    fwd_variance_word : jnp.ndarray
        Forward variances for a single word, shape (num_time_slices + 1,)
    time_obs_idx : int
        Index of the observation time for which to compute the derivative
    chain_variance : float
        Variance parameter for the state transition
    obs_variance : float
        Variance parameter for the observations
    num_time_slices : int
        Number of time slices
        
    Returns
    -------
    jnp.ndarray
        Derivative array of shape (num_time_slices + 1,)
    """
    T = num_time_slices
    
    # Initialize derivative array
    deriv = jnp.zeros(T + 1)
    
    # Forward pass
    for t in range(1, T + 1):
        denominator = fwd_variance_word[t - 1] + chain_variance + obs_variance
        w_factor = jnp.where(
            (obs_variance > 0.0) & (denominator != 0.0),
            obs_variance / denominator,
            0.0
        )
        
        val = w_factor * deriv[t - 1]
        if time_obs_idx == (t - 1):
            val += 1.0 - w_factor
            
        deriv = deriv.at[t].set(val)
    
    # Backward pass
    for t in range(T - 1, -1, -1):
        fwd_var_s_plus_chain_var = fwd_variance_word[t] + chain_variance
        w_factor_backward = jnp.where(
            (chain_variance != 0.0) & (fwd_var_s_plus_chain_var != 0.0),
            chain_variance / fwd_var_s_plus_chain_var,
            0.0
        )
        
        deriv = deriv.at[t].set(
            w_factor_backward * deriv[t] + (1.0 - w_factor_backward) * deriv[t + 1]
        )
    
    return deriv

def jax_compute_obs_deriv(
    current_mean_word, current_variance_word, zeta_topic, word_counts_w,
    totals_time, mean_deriv_mtx_w, chain_variance, num_time_slices
):
    """JAX implementation of compute_obs_deriv.
    
    Computes gradient d(objective_func) / d(obs_{t,w}) for t = 0..T-1.
    
    Parameters
    ----------
    current_mean_word : jnp.ndarray
        Current mean values for a single word, shape (num_time_slices + 1,)
    current_variance_word : jnp.ndarray
        Current variance values for a single word, shape (num_time_slices + 1,)
    zeta_topic : jnp.ndarray
        Zeta values for the topic, shape (num_time_slices,)
    word_counts_w : jnp.ndarray
        Word counts for each time slice, shape (num_time_slices,)
    totals_time : jnp.ndarray
        Total counts for each time slice, shape (num_time_slices,)
    mean_deriv_mtx_w : jnp.ndarray
        Matrix of mean derivatives, shape (num_time_slices, num_time_slices + 1)
    chain_variance : float
        Variance parameter for the state transition
    num_time_slices : int
        Number of time slices
        
    Returns
    -------
    jnp.ndarray
        Gradient array of shape (num_time_slices,)
    """
    T = num_time_slices
    
    # Initialize gradient array
    deriv_output = jnp.zeros(T)
    
    # Precompute temp_vect: exp(mean_w[u+1] + var_w[u+1]/2) for u=0..T-1
    temp_vect_topic_w = jnp.exp(
        current_mean_word[1:T+1] + current_variance_word[1:T+1] / 2.0
    )
    
    for t_obs_idx in range(T):  # t_obs_idx for d/d(obs_t), from 0 to T-1
        mean_deriv_for_obs_t = mean_deriv_mtx_w[t_obs_idx, :]  # Shape (T+1,)
        deriv_val_for_obs_t = 0.0
        
        # Part 1: Derivative of sum_{u=0..T-1} [ wc_w[u]*m_w[u+1] - totals[u]*exp(m_w[u+1]+v_w[u+1]/2)/zeta_topic[u] ]
        for u_time_idx in range(T):  # u_time_idx from 0 to T-1
            dmean_u_plus_1_dobs_t = mean_deriv_for_obs_t[u_time_idx + 1]
            
            # Word count term
            deriv_val_for_obs_t += word_counts_w[u_time_idx] * dmean_u_plus_1_dobs_t
            
            # Zeta term
            exp_term_w_u = temp_vect_topic_w[u_time_idx]
            zeta_u = zeta_topic[u_time_idx]
            
            if zeta_u > 0.0:
                # Simplified derivative of (exp_term_w / zeta_u) w.r.t mean_w[u+1]
                factor = (zeta_u - exp_term_w_u) / (zeta_u * zeta_u)
                deriv_val_for_obs_t -= (
                    totals_time[u_time_idx]
                    * exp_term_w_u
                    * dmean_u_plus_1_dobs_t
                    * factor
                )
        
        # Part 2: Derivative of sum_{u=0..T-1} [ - ( (m_w[u+1]-m_w[u])^2/(2cv) - v_w[u+1]/cv - log(cv) ) ]
        for u_time_idx in range(T):
            mean_u_plus_1 = current_mean_word[u_time_idx + 1]
            mean_u = current_mean_word[u_time_idx]
            dmean_u_plus_1_dobs_t = mean_deriv_for_obs_t[u_time_idx + 1]
            dmean_u_dobs_t = mean_deriv_for_obs_t[u_time_idx]
            
            if chain_variance > 0.0:
                deriv_val_for_obs_t -= (
                    (mean_u_plus_1 - mean_u)
                    / chain_variance
                    * (dmean_u_plus_1_dobs_t - dmean_u_dobs_t)
                )
        
        # Part 3: Prior term for m_w[0]
        if chain_variance > 0.0 and INIT_MULT > 0.0:
            deriv_val_for_obs_t -= (
                current_mean_word[0]
                / (INIT_MULT * chain_variance)
                * mean_deriv_for_obs_t[0]
            )
        
        deriv_output = deriv_output.at[t_obs_idx].set(deriv_val_for_obs_t)
    
    return deriv_output

def jax_f_obs(
    x_obs_w, word_counts_w, totals_time, variance_word_fixed, fwd_variance_word_fixed,
    chain_variance, obs_variance_scalar, num_time_slices, zeta_topic_fixed,
    mean_word_buffer, fwd_mean_word_buffer
):
    """JAX implementation of f_obs.
    
    Objective function for optimizing obs values.
    
    Parameters
    ----------
    x_obs_w : jnp.ndarray
        Current obs values for a single word, shape (num_time_slices,)
    word_counts_w : jnp.ndarray
        Word counts for each time slice, shape (num_time_slices,)
    totals_time : jnp.ndarray
        Total counts for each time slice, shape (num_time_slices,)
    variance_word_fixed : jnp.ndarray
        Fixed variance values for a single word, shape (num_time_slices + 1,)
    fwd_variance_word_fixed : jnp.ndarray
        Fixed forward variance values for a single word, shape (num_time_slices + 1,)
    chain_variance : float
        Variance parameter for the state transition
    obs_variance_scalar : float
        Variance parameter for the observations
    num_time_slices : int
        Number of time slices
    zeta_topic_fixed : jnp.ndarray
        Fixed zeta values for the topic, shape (num_time_slices,)
    mean_word_buffer : jnp.ndarray
        Buffer for storing mean values, shape (num_time_slices + 1,)
    fwd_mean_word_buffer : jnp.ndarray
        Buffer for storing forward mean values, shape (num_time_slices + 1,)
        
    Returns
    -------
    float
        Objective function value (negative for minimization)
    """
    T = num_time_slices
    
    # Compute mean based on current x_obs_w
    mean, _ = jax_compute_post_mean_scan_unjitted(
        x_obs_w, fwd_variance_word_fixed, chain_variance, obs_variance_scalar, T
    )
    
    # Convert to numpy for compatibility with existing code
    mean_np = np.array(mean)
    
    # Objective terms
    current_objective_val = 0.0
    
    for t_idx in range(T):
        m_wt_plus_1 = mean_np[t_idx + 1]
        m_wt = mean_np[t_idx]
        
        # Word count term
        current_objective_val += word_counts_w[t_idx] * m_wt_plus_1
        
        # Zeta term
        if zeta_topic_fixed[t_idx] > 0.0:
            current_objective_val -= (
                totals_time[t_idx]
                * np.exp(m_wt_plus_1 + variance_word_fixed[t_idx + 1] / 2.0)
                / zeta_topic_fixed[t_idx]
            )
        
        # Chain variance term
        if chain_variance > 0.0:
            chain_term_penalty = (
                (np.power(m_wt_plus_1 - m_wt, 2) / (2 * chain_variance))
                - (variance_word_fixed[t_idx + 1] / chain_variance)
                - np.log(chain_variance)
            )
            current_objective_val -= chain_term_penalty
    
    # Prior term for m_w[0]
    if chain_variance > 0.0 and INIT_MULT > 0.0:
        current_objective_val -= np.power(mean_np[0], 2) / (
            2 * INIT_MULT * chain_variance
        )
    
    return -current_objective_val  # Negative for minimization

def jax_df_obs(
    x_obs_w, word_counts_w, totals_time, variance_word_fixed, fwd_variance_word_fixed,
    chain_variance, obs_variance_scalar, num_time_slices, zeta_topic_fixed,
    mean_word_buffer, fwd_mean_word_buffer, mean_deriv_mtx_w, grad_output_buffer
):
    """JAX implementation of df_obs.
    
    Gradient function for optimizing obs values.
    
    Parameters
    ----------
    x_obs_w : jnp.ndarray
        Current obs values for a single word, shape (num_time_slices,)
    word_counts_w : jnp.ndarray
        Word counts for each time slice, shape (num_time_slices,)
    totals_time : jnp.ndarray
        Total counts for each time slice, shape (num_time_slices,)
    variance_word_fixed : jnp.ndarray
        Fixed variance values for a single word, shape (num_time_slices + 1,)
    fwd_variance_word_fixed : jnp.ndarray
        Fixed forward variance values for a single word, shape (num_time_slices + 1,)
    chain_variance : float
        Variance parameter for the state transition
    obs_variance_scalar : float
        Variance parameter for the observations
    num_time_slices : int
        Number of time slices
    zeta_topic_fixed : jnp.ndarray
        Fixed zeta values for the topic, shape (num_time_slices,)
    mean_word_buffer : jnp.ndarray
        Buffer for storing mean values, shape (num_time_slices + 1,)
    fwd_mean_word_buffer : jnp.ndarray
        Buffer for storing forward mean values, shape (num_time_slices + 1,)
    mean_deriv_mtx_w : jnp.ndarray
        Matrix for storing mean derivatives, shape (num_time_slices, num_time_slices + 1)
    grad_output_buffer : jnp.ndarray
        Buffer for storing gradient values, shape (num_time_slices,)
        
    Returns
    -------
    jnp.ndarray
        Gradient array of shape (num_time_slices,)
    """
    T = num_time_slices
    
    # Compute mean based on current x_obs_w
    mean, _ = jax_compute_post_mean_scan_unjitted(
        x_obs_w, fwd_variance_word_fixed, chain_variance, obs_variance_scalar, T
    )
    
    # Convert to numpy for compatibility with existing code
    mean_np = np.array(mean)
    
    # Compute mean derivatives for each observation time
    mean_deriv_mtx = np.zeros((T, T + 1))
    for t in range(T):
        deriv = jax_compute_mean_deriv(
            fwd_variance_word_fixed, t, chain_variance, obs_variance_scalar, T
        )
        mean_deriv_mtx[t] = np.array(deriv)
    
    # Compute gradient
    grad = jax_compute_obs_deriv(
        mean, variance_word_fixed, zeta_topic_fixed,
        word_counts_w, totals_time, mean_deriv_mtx,
        chain_variance, T
    )
    
    # Convert to numpy for compatibility with existing code
    grad_np = np.array(grad)
    
    return -grad_np  # Negative for minimization
