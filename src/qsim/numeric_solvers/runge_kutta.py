import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def rungeKutta(rhs_func, psi0, t_start, t_end, dt, coeffs, trajectory):
    math_dt = abs(dt)
    n_steps = trajectory.shape[0] - 1
    
    # 1. Precompute constants outside the loop
    dt_2 = math_dt / 2.0
    dt_6 = math_dt / 6.0
    
    # 2. Pre-allocate temporary arrays for RK4 stages
    k1 = np.empty_like(psi0)
    k2 = np.empty_like(psi0)
    k3 = np.empty_like(psi0)
    k4 = np.empty_like(psi0)
    psi_temp = np.empty_like(psi0)
    
    # Flatten views for guaranteed zero-allocation fast loops
    k1_f = k1.ravel()
    k2_f = k2.ravel()
    k3_f = k3.ravel()
    k4_f = k4.ravel()
    psi_temp_f = psi_temp.ravel()
    
    # Set initial state
    trajectory[0] = psi0
    size = psi0.size
    
    for i in range(n_steps):
        c_t = coeffs[2*i]
        c_half = coeffs[2*i + 1]
        c_next = coeffs[2*i + 2]
        
        # Current state view
        psi_curr = trajectory[i]
        psi_curr_f = psi_curr.ravel()
        
        # --- Stage 1 ---
        rhs_func(psi_curr, c_t, k1)
        for j in range(size):
            psi_temp_f[j] = psi_curr_f[j] + dt_2 * k1_f[j]
            
        # --- Stage 2 ---
        rhs_func(psi_temp, c_half, k2)
        for j in range(size):
            psi_temp_f[j] = psi_curr_f[j] + dt_2 * k2_f[j]
            
        # --- Stage 3 ---
        rhs_func(psi_temp, c_half, k3)
        for j in range(size):
            psi_temp_f[j] = psi_curr_f[j] + math_dt * k3_f[j]
            
        # --- Stage 4 ---
        rhs_func(psi_temp, c_next, k4)
        
        # --- Final Update ---
        psi_next = trajectory[i+1]
        psi_next_f = psi_next.ravel()
        for j in range(size):
            psi_next_f[j] = psi_curr_f[j] + dt_6 * (k1_f[j] + 2.0*k2_f[j] + 2.0*k3_f[j] + k4_f[j])
            
    return trajectory
