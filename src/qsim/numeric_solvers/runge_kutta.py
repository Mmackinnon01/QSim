import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def rungeKutta(rhs_func, psi0, dt, coeffs, trajectory):
    math_dt = abs(dt)
    n_steps = trajectory.shape[0] - 1
    size = psi0.size
    
    dt_2 = math_dt / 2.0
    dt_6 = math_dt / 6.0
    
    # 1. Pre-allocate 2D arrays (these get passed to rhs_func)
    k1 = np.empty_like(psi0)
    k2 = np.empty_like(psi0)
    k3 = np.empty_like(psi0)
    k4 = np.empty_like(psi0)
    
    psi_temp = np.empty_like(psi0)
    
    # 2. Create 1D "shadow views" outside the loop.
    # Because these share memory with the 2D arrays, modifying them 
    # instantly modifies the 2D versions without ANY allocation overhead.
    k1_f = k1.reshape(size)
    k2_f = k2.reshape(size)
    k3_f = k3.reshape(size)
    k4_f = k4.reshape(size)
    psi_temp_f = psi_temp.reshape(size)
    
    # Create a shadow view of the entire trajectory
    traj_flat = trajectory.reshape((trajectory.shape[0], size))
    
    # Set initial state
    trajectory[0] = psi0
    
    for i in range(n_steps):
        c_t = coeffs[2*i]
        c_half = coeffs[2*i + 1]
        c_next = coeffs[2*i + 2]
        
        # 3. Get both the 2D view (for rhs_func) and 1D view (for math)
        psi_curr = trajectory[i]
        psi_curr_f = traj_flat[i]
        
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
        psi_next_f = traj_flat[i+1]
        for j in range(size):
            psi_next_f[j] = psi_curr_f[j] + dt_6 * (k1_f[j] + 2.0*k2_f[j] + 2.0*k3_f[j] + k4_f[j])