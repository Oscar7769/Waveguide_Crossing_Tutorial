import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.ndimage 

from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import dimod
from dwave.samplers import SimulatedAnnealingSampler

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Warning: dwave-ocean-sdk not installed. QA mode will fail.")

import meep as mp
from factorization_machine import FactorizationMachine

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

WORK_TAG = 1
STOP_TAG = 2

RESOLUTION = 40
DPML = 1.0
MDM_LC = 2.4      # 設計區域大小
WG_LENGTH = 1.0   # 波導長度
WG_WIDTH = 1.0    # 波導寬度

SX = 2 * DPML + MDM_LC + 2 * WG_LENGTH
SY = 2 * DPML + MDM_LC + 2 * WG_LENGTH
CELL = mp.Vector3(SX, SY, 0)

N_SIO2 = 1.44
N_SI = 3.48
SIO2_MEDIUM = mp.Medium(index=N_SIO2)
SI_MEDIUM = mp.Medium(index=N_SI)

wl_cen = 1.55 
FCEN = 1 / wl_cen
DF = 0.1 * FCEN
NFREQ = 1

KERNEL_SIZE = 5
KERNEL_SIGMA = 1.00
TANH_BETA = 50
TANH_ETA = 0.5
SMOOTH_THRESHOLD = 0.5
NUM_SWEEPS = 1000


def gaussian_kernel(size=KERNEL_SIZE, sigma=KERNEL_SIGMA):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def tanh_projection(x, beta=TANH_BETA, eta=TANH_ETA):
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den

def enforce_reflection_symmetry(matrix_quadrant):
    Q = np.array(matrix_quadrant)
    Q_upper = np.triu(Q)
    Q_symmetric = Q_upper + np.triu(Q, 1).T
    top_half = np.hstack((Q_symmetric, np.fliplr(Q_symmetric)))
    bottom_half = np.hstack((np.flipud(Q_symmetric), np.flipud(np.fliplr(Q_symmetric))))
    
    return np.vstack((top_half, bottom_half))

def get_projected_density_matrix(binary_vector, grid_rows, grid_cols):
    vec_len = len(binary_vector)
    if vec_len == grid_rows * grid_cols:
        grid_matrix = np.array(binary_vector).astype(float).reshape((grid_rows, grid_cols))
    elif vec_len == (grid_rows // 2) * (grid_cols // 2):
        q_rows = grid_rows // 2
        q_cols = grid_cols // 2
        Q = np.array(binary_vector).astype(float).reshape((q_rows, q_cols))
        grid_matrix = enforce_reflection_symmetry(Q)
    else:
        raise ValueError(f"Invalid binary_vector size: {vec_len}")

    kernel = gaussian_kernel(size=KERNEL_SIZE, sigma=KERNEL_SIGMA)
    try:
        from scipy.signal import convolve2d
        density = convolve2d(grid_matrix, kernel, mode='same', boundary='symm')
    except Exception:
        k = kernel.shape[0]
        pad = k // 2
        img_p = np.pad(grid_matrix, ((pad, pad), (pad, pad)), mode='reflect')
        density = np.zeros_like(grid_matrix)
        for i in range(grid_rows):
            for j in range(grid_cols):
                density[i, j] = np.sum(img_p[i:i+k, j:j+k] * kernel)
    
    projected_density = tanh_projection(density, beta=TANH_BETA, eta=TANH_ETA)
    projected_density = np.clip(projected_density, 0.0, 1.0)
    return projected_density

def create_projected_geometry(binary_vector, grid_rows, grid_cols):
    density = get_projected_density_matrix(binary_vector, grid_rows, grid_cols)
    weights = density.flatten()
    material_grid = mp.MaterialGrid(mp.Vector3(grid_cols, grid_rows), SIO2_MEDIUM, SI_MEDIUM, weights=weights)
    material_grid.smoothing_radius = 1.0 
    design_block = mp.Block(size=mp.Vector3(MDM_LC, MDM_LC, mp.inf), center=mp.Vector3(), material=material_grid)
    return [design_block]

def generate_smooth_random_config(rows, cols):
    small_r, small_c = max(1, (rows + 1) // 2), max(1, (cols + 1) // 2) 
    noise = np.random.rand(small_r, small_c)
    smooth_noise = scipy.ndimage.zoom(noise, zoom=2.0, order=1) 
    smooth_noise = smooth_noise[:rows, :cols]
    binary = (smooth_noise > SMOOTH_THRESHOLD).astype(int).flatten()
    return binary

def evaluate_mdm_mode(binary_vector, grid_rows, grid_cols, mode_name):
    mp.Simulation(cell_size=CELL, resolution=1, boundary_layers=[]).reset_meep()
    mdm_structure = create_projected_geometry(binary_vector, grid_rows, grid_cols)
    
    input_wg_center_x = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2   
    through_wg_center_x = MDM_LC / 2 + (WG_LENGTH + DPML) / 2   
    cross_top_center_y = MDM_LC / 2 + (WG_LENGTH + DPML) / 2    
    cross_bot_center_y = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2   
    
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(input_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(through_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(0, cross_top_center_y), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(0, cross_bot_center_y), material=SI_MEDIUM),
    ]
    
    full_geometry = fixed_geometry + mdm_structure
    
    src_center = mp.Vector3(-SX / 2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH) 
    mon_x_through = MDM_LC / 2 + WG_LENGTH / 2
    mon_y_cross_top = MDM_LC / 2 + WG_LENGTH / 2
    mon_y_cross_bot = -MDM_LC / 2 - WG_LENGTH / 2
    monitor_size_y = mp.Vector3(0, WG_WIDTH * 5)
    monitor_size_x = mp.Vector3(WG_WIDTH * 5, 0)
    
    mode_props = {
        'TE0': {'band_num': 1, 'parity': mp.EVEN_Y, 'global_band': 1},
        'TE1': {'band_num': 2, 'parity': mp.ODD_Y, 'global_band': 2}
    }
    props = mode_props[mode_name]
    
    sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                  center=src_center, size=src_size, direction=mp.X, 
                                  eig_band=props['band_num'], eig_parity=props['parity'])]

    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], 
                        geometry=full_geometry, sources=sources, 
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)

    target_wls = [1.52, 1.56, 1.60]
    target_freqs = [1/wl for wl in target_wls]

    in_flux_region = mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=monitor_size_y)
    norm_fluxes = [sim.add_mode_monitor(f, 0, 1, in_flux_region) for f in target_freqs]
    
    through_flux_region = mp.ModeRegion(center=mp.Vector3(mon_x_through, 0), size=monitor_size_y)
    flux_throughs = [sim.add_mode_monitor(f, 0, 1, through_flux_region) for f in target_freqs]
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(mon_x_through, WG_WIDTH/4), 1e-4))
    
    T_through_list = []
    for i in range(len(target_freqs)):
        input_flux = sim.get_eigenmode_coefficients(norm_fluxes[i], [props['band_num']], eig_parity=props['parity']).alpha[0, 0, 0]
        input_power = np.abs(input_flux)**2 + 1e-12
        
        through_coeff = sim.get_eigenmode_coefficients(flux_throughs[i], [props['band_num']], eig_parity=props['parity']).alpha[0, 0, 0]
        T_through = np.abs(through_coeff)**2 / input_power
        T_through_list.append(T_through)

    sim.reset_meep()
    return {'through': T_through_list}

def plot_transmission_spectrum(wls, trans_dbs, mode_name, output_folder):
    from scipy.interpolate import make_interp_spline
    plt.figure(figsize=(6, 5))
    wls = np.array(wls)
    trans_dbs = np.array(trans_dbs)
    sorted_idx = np.argsort(wls)
    wls = wls[sorted_idx]
    trans_dbs = trans_dbs[sorted_idx]
    
    if len(wls) >= 3:
        x_smooth = np.linspace(wls.min(), wls.max(), 100)
        spline = make_interp_spline(wls, trans_dbs, k=2)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='FMQA')
    else:
        plt.plot(wls, trans_dbs, 'b-', linewidth=2, label='FMQA')
        
    plt.title(f"Transmission Spectrum ({mode_name})", fontsize=16)
    plt.xlabel(r"wavelength ($\mu$m)", fontsize=14)
    plt.ylabel("Transmission (dB)", fontsize=14)
    plt.ylim([-2, 0])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Transmission_Spectrum_{mode_name}.png"), dpi=300)
    plt.close()

def perform_detailed_final_analysis(best_config, grid_rows, grid_cols, output_folder):
    print(f"\n>>> Starting Final Detailed MEEP Analysis (TE0 Waveguide Crossing) <<<")
    

    mdm_structure = create_projected_geometry(best_config, grid_rows, grid_cols)
    
    input_wg_center_x = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2    
    through_wg_center_x = MDM_LC / 2 + (WG_LENGTH + DPML) / 2   
    cross_top_center_y = MDM_LC / 2 + (WG_LENGTH + DPML) / 2    
    cross_bot_center_y = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2   
    
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(input_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(through_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(0, cross_top_center_y), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(0, cross_bot_center_y), material=SI_MEDIUM),
    ]
    
    full_geometry = fixed_geometry + mdm_structure
    
    src_center = mp.Vector3(-SX / 2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH) 
    mon_x_through = MDM_LC / 2 + WG_LENGTH / 2
    mon_y_cross_top = MDM_LC / 2 + WG_LENGTH / 2
    mon_y_cross_bot = -MDM_LC / 2 - WG_LENGTH / 2
    monitor_size_y = mp.Vector3(0, WG_WIDTH * 3)
    monitor_size_x = mp.Vector3(WG_WIDTH * 5, 0) 
    
    mode_definitions = ['TE0', 'TE1']
    detailed_results = {}
    
    for mode_name in mode_definitions:
        mode_props = {
            'TE0': {'band_num': 1, 'parity': mp.EVEN_Y, 'global_band': 1},
            'TE1': {'band_num': 2, 'parity': mp.ODD_Y, 'global_band': 2}
        }
        props = mode_props[mode_name]
        
        sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                      center=src_center, size=src_size, direction=mp.X, 
                                      eig_band=props['band_num'], eig_parity=props['parity'])]
    
        sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], 
                        geometry=full_geometry, sources=sources, 
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)
        
        target_wls = [1.52, 1.56, 1.60]
        target_freqs = [1/wl for wl in target_wls]

        in_flux_region = mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=monitor_size_y)
        norm_fluxes = [sim.add_mode_monitor(f, 0, 1, in_flux_region) for f in target_freqs]
        
        through_flux_region = mp.ModeRegion(center=mp.Vector3(mon_x_through, 0), size=monitor_size_y)
        flux_throughs = [sim.add_mode_monitor(f, 0, 1, through_flux_region) for f in target_freqs]
        
        dft_monitor = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(), size=mp.Vector3(SX, SY))
    
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(mon_x_through, WG_WIDTH/4), 1e-4))
        
        ez_dft_data = sim.get_dft_array(dft_monitor, mp.Ez, 0)
        eps_data = sim.get_epsilon()
        intensity = np.abs(ez_dft_data)**2
        
        T_through_list = []
        for i in range(len(target_freqs)):
            res_input = sim.get_eigenmode_coefficients(norm_fluxes[i], [props['band_num']], eig_parity=props['parity']).alpha[0, 0, 0]
            input_power = np.abs(res_input)**2 + 1e-12
            
            through_coeff = sim.get_eigenmode_coefficients(flux_throughs[i], [props['band_num']], eig_parity=props['parity']).alpha[0, 0, 0]
            T_through = np.abs(through_coeff)**2 / input_power
            T_through_list.append(T_through)

        detailed_results[mode_name] = {}
        trans_dbs_plot = []
        for idx, wl in enumerate(target_wls):
            trans = T_through_list[idx]
            trans_db = 10 * np.log10(trans + 1e-9)
            trans_dbs_plot.append(trans_db)
            print(f"  [Result] {mode_name} L->R (Through) @ {wl}$\mu$m: {trans:.4f} ({trans_db:.2f} dB)")
            detailed_results[mode_name][f"{wl}$\mu$m"] = [float(trans), float(trans_db)]
            
        plot_transmission_spectrum(target_wls, trans_dbs_plot, mode_name, output_folder)

        x = np.linspace(-SX/2, SX/2, intensity.shape[0])
        y = np.linspace(-SY/2, SY/2, intensity.shape[1])
    
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(intensity.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   cmap='inferno', origin='lower')
        ax.contour(eps_data.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.5, linewidths=1, origin='lower')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label=r'Intensity $|E_z|^2$')
        ax.set_title(f"Optimized Final Structure ({mode_name})", fontsize=16)
        ax.set_xlabel(r"x ($\mu$m)", fontsize=14)
        ax.set_ylabel(r"y ($\mu$m)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"Optimized_Final_Structure_{mode_name}.png"), dpi=300)
        plt.close(fig)

    x_mask = (x >= -MDM_LC/2) & (x <= MDM_LC/2)
    y_mask = (y >= -MDM_LC/2) & (y <= MDM_LC/2)
    eps_design = eps_data[np.ix_(x_mask, y_mask)]
    
    threshold_eps = (N_SI**2 + N_SIO2**2) / 2
    binary_design = (eps_design > threshold_eps).astype(int)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(binary_design.T, extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2], 
              cmap='gray_r', origin='lower')
    ax.set_title("Smoothed Binary Structure", fontsize=16)
    ax.set_xlabel(r"x ($\mu$m)", fontsize=14)
    ax.set_ylabel(r"y ($\mu$m)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Smoothed_Binary_Structure.png"), dpi=300)
    plt.close(fig)

    return detailed_results

def plot_fom_history(fom_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FOM vs Iteration", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Best FOM", fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fom_evolution.png"), dpi=300)
    plt.close()

def plot_optimization_trajectory(foms, split_index, adding_num, output_folder):
    plt.figure(figsize=(12, 6))
    
    iterations = np.zeros(len(foms))
    iterations[:split_index] = 0
    if len(foms) > split_index:
        opt_indices = np.arange(len(foms) - split_index)
        iterations[split_index:] = (opt_indices // adding_num) + 1
    
    plt.scatter(iterations[:split_index], foms[:split_index], 
                s=5, c='red', alpha=0.6, label='Initial Random Samples')
                
    if len(foms) > split_index:
        plt.scatter(iterations[split_index:], foms[split_index:], 
                    s=5, c='blue', alpha=0.6, label='FMQA Optimized Samples')
    
    if len(foms) > split_index:
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Opt Start')
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Optimization FOM', fontsize=14)
    plt.title('Optimization Trajectory', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ax = plt.gca()
    max_iter = int(np.max(iterations))
    if max_iter > 0:
        step = max(1, max_iter // 10)
        ax.set_xticks(np.arange(0, max_iter + 1, step))
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "fom_trajectory.png"), dpi=150)
    plt.close()

def plot_time_statistics_pie_chart(total_fm, total_sa, total_fdtd, other, output_folder):
    labels = ['FM Training', 'Annealing Sampling', 'FDTD Simulation', 'Other']
    sizes = [total_fm, total_sa, total_fdtd, max(0, other)]
    all_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    total = sum(sizes)
    filtered_sizes = []
    filtered_labels = []
    filtered_colors = []
    for s, l, c in zip(sizes, labels, all_colors):
        if s > 0.1:
            filtered_sizes.append(s)
            pct = (s / total) * 100 if total > 0 else 0
            filtered_labels.append(f"{l} ({pct:.1f}%)")
            filtered_colors.append(c)
            
    fig, ax = plt.subplots(figsize=(10, 8))
    explode = [0.01] * len(filtered_sizes)
    
    def my_autopct(pct):
        return f'{pct:.1f}%' if pct > 3 else ''
        
    wedges, texts, autotexts = ax.pie(filtered_sizes, 
                                      autopct=my_autopct, 
                                      startangle=140, 
                                      colors=filtered_colors,
                                      explode=explode,
                                      textprops={'fontsize': 16})
    plt.setp(autotexts, size=16, weight="bold")
    
    lgd = ax.legend(wedges, filtered_labels, title="Runtime Components", 
                    loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=14, title_fontsize=16)

    ax.set_title("Total Runtime Distribution", fontsize=18)
    plt.savefig(os.path.join(output_folder, "time_distribution.png"), 
                dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def train_fm_model(model, X_train, Y_train, num_epoch, learning_rate, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    device = next(model.parameters()).device
    
    X_tensor = torch.from_numpy(X_train).float().to(device)
    y_mean = Y_train.mean()
    y_std = Y_train.std() + 1e-8
    Y_scaled = (Y_train - y_mean) / y_std
    Y_tensor = torch.from_numpy(Y_scaled).float().view(-1, 1).to(device)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    start_time = time.time()
    model.train() 
    
    print(f"  [FM Training] Dataset Size: {len(X_train)} samples")
    
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    - Epoch [{epoch+1:3d}/{num_epoch}], Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
    return model, loss_history, time.time() - start_time, (y_mean, y_std)

def worker_node():
    grid_dims = comm.bcast(None, root=0)
    mp.verbosity(0)
    while True:
        status = MPI.Status()
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG: break
        task_idx, config, mode_name = data
        res = evaluate_mdm_mode(config, grid_dims['rows'], grid_dims['cols'], mode_name)
        comm.send((task_idx, mode_name, res), dest=0, tag=WORK_TAG)

def parallel_evaluate(tasks, fom_cache, grid_rows, grid_cols):
    num_tasks = len(tasks)
    results = [None] * num_tasks
    partial_results = [{} for _ in range(num_tasks)]
    tasks_to_run = []
    
    for i, config in enumerate(tasks):
        config_tuple = tuple(config)
        if config_tuple in fom_cache: 
            results[i] = fom_cache[config_tuple]
        else: 
            tasks_to_run.append((i, config, 'TE0'))
            tasks_to_run.append((i, config, 'TE1'))

    if not tasks_to_run: return results

    num_to_run = len(tasks_to_run)
    sent_jobs = 0
    jobs_done = 0
    for worker_rank in range(1, min(size, num_to_run + 1)):
        comm.send(tasks_to_run[sent_jobs], dest=worker_rank, tag=WORK_TAG)
        sent_jobs += 1

    while jobs_done < num_to_run:
        status = MPI.Status()
        task_idx, mode_name, res_dict = comm.recv(source=MPI.ANY_SOURCE, tag=WORK_TAG, status=status)
        partial_results[task_idx][mode_name] = res_dict
        
        if 'TE0' in partial_results[task_idx] and 'TE1' in partial_results[task_idx]:
            trans_res = partial_results[task_idx]
            
            TE0_T = np.array(trans_res['TE0']['through'])
            TE1_T = np.array(trans_res['TE1']['through'])
            avg_T = np.mean(np.concatenate([TE0_T, TE1_T]))
            fom = 1.0 - avg_T
            
            results[task_idx] = fom
            fom_cache[tuple(tasks[task_idx])] = fom
            
        jobs_done += 1
        if sent_jobs < num_to_run:
            comm.send(tasks_to_run[sent_jobs], dest=status.Get_source(), tag=WORK_TAG)
            sent_jobs += 1
            
    return results

def master_node():
    PARAMS = {
        'GRID_ROWS': 12,
        'GRID_COLS': 12,
        'INIT_DATASET_SIZE': 1000,
        'ITERATIONS': 100,
        'ADDING_NUM': 30,
        'NUM_EPOCHS': 1000,
        'LEARNING_RATE': 1.0e-3,
        'NUM_READS': 1500,
        'K_FACTOR': 8,
        'SAMPLER_TYPE': 'SA'
    }
    parser = argparse.ArgumentParser(description="FMQA Crossing Inverse Design")
    parser.add_argument('--name', type=str, default=f'Crossing_waveguide_{PARAMS["INIT_DATASET_SIZE"]}_{PARAMS["ITERATIONS"]}x{PARAMS["ADDING_NUM"]}', help='Job Name')
    args = parser.parse_args()
    job_name = args.name
    total_meep_time = 0.0
    start_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Master Node Started on {device} (Job: {job_name}) ---")

    Q_ROWS = PARAMS['GRID_ROWS']
    Q_COLS = PARAMS['GRID_COLS']
    FULL_ROWS = Q_ROWS * 2
    FULL_COLS = Q_COLS * 2
    NUM_VARS = Q_ROWS * Q_COLS
    
    BASE_PATH = os.getcwd() 
    RESULTS_BASE_DIR = os.path.join(BASE_PATH, "Results_Crossing")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{job_name}_{timestamp}"
    output_folder = os.path.join(RESULTS_BASE_DIR, run_dir_name)
    os.makedirs(output_folder, exist_ok=True)
    
    comm.bcast({'rows': FULL_ROWS, 'cols': FULL_COLS}, root=0)
    
    print(f"--- Generating Initial Dataset ({PARAMS['INIT_DATASET_SIZE']}) ---")
    configs = np.array([
        generate_smooth_random_config(Q_ROWS, Q_COLS) 
        for _ in range(PARAMS['INIT_DATASET_SIZE'])
    ])
    
    fom_cache = {}
    t_init_meep_start = time.time()
    foms = np.array(parallel_evaluate(configs, fom_cache, FULL_ROWS, FULL_COLS))
    for i, c in enumerate(configs): fom_cache[tuple(c)] = foms[i]
    total_meep_time += time.time() - t_init_meep_start
    
    best_fom = np.min(foms)
    best_config = configs[np.argmin(foms)]
    print(f"Initial Best FOM: {best_fom:.4f}")

    history = {'chain_break_trends': {'avg': [], 'max': []}, 'timing_metrics': {'fm_train_time': [], 'new_data_sim_time': [], 'sa_time': []}}
    best_fom_history = [best_fom] 

    model = FactorizationMachine(input_size=NUM_VARS, factorization_size=PARAMS['K_FACTOR']).to(device)

    for i in range(PARAMS['ITERATIONS']):
        print(f"\n=== Iteration {i+1}/{PARAMS['ITERATIONS']} ===")
        
        model, _, t_train, _ = train_fm_model(model, configs, foms, PARAMS['NUM_EPOCHS'], PARAMS['LEARNING_RATE'])
        history['timing_metrics']['fm_train_time'].append(t_train)
        
        bias, h, Q = model.get_bhQ() 
        Q_dict = {(r, c): Q[r, c] for r in range(Q.shape[0]) for c in range(r+1, Q.shape[1]) if Q[r, c] != 0}
        bqm = dimod.BinaryQuadraticModel(h, Q_dict, bias, dimod.BINARY)
        
        t_sa_start = time.time()
        sampleset = None
        if PARAMS['SAMPLER_TYPE'] == "QA" and DWAVE_AVAILABLE:
            try:
                sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], label=f"{job_name}_{i}")
                if 'chain_break_fraction' in sampleset.record.dtype.names:
                     history['chain_break_trends']['avg'].append(float(np.mean(sampleset.record['chain_break_fraction'])))
            except:
                print("QA failed, using SA")
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], num_sweeps=NUM_SWEEPS)
        else:
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], num_sweeps=NUM_SWEEPS)
            
        sampleset = sampleset.aggregate()
        sample_configs = sampleset.record['sample'][np.argsort(sampleset.record['energy'])]
        history['timing_metrics']['sa_time'].append(time.time() - t_sa_start)
        
        unique_new = [s for s in sample_configs if tuple(s) not in {tuple(x) for x in configs}]
        new_configs_list = unique_new[:PARAMS['ADDING_NUM']]
        
        shortfall = PARAMS['ADDING_NUM'] - len(new_configs_list)
        if shortfall > 0:
            print(f"  [Notice] Sampler only yielded {len(new_configs_list)} unique configs. Auto-filling {shortfall} random configs.")
            existing_set = {tuple(x) for x in configs} | {tuple(x) for x in new_configs_list}
            attempts = 0
            while len(new_configs_list) < PARAMS['ADDING_NUM'] and attempts < shortfall * 10:
                rand_c = generate_smooth_random_config(Q_ROWS, Q_COLS)
                if tuple(rand_c) not in existing_set:
                    new_configs_list.append(rand_c)
                    existing_set.add(tuple(rand_c))
                attempts += 1
                
        new_configs = np.array(new_configs_list)
        
        if new_configs.size > 0:
            num_new_samples = len(new_configs)
            print(f"  Evaluating {num_new_samples} new candidates...")
            
            t_sim_start = time.time()
            new_foms = parallel_evaluate(new_configs, fom_cache, FULL_ROWS, FULL_COLS)
            sim_duration = time.time() - t_sim_start
            total_meep_time += sim_duration
            configs = np.vstack([configs, new_configs])
            foms = np.concatenate([foms, new_foms])
            history['timing_metrics']['new_data_sim_time'].append(time.time() - t_sim_start)
            
            curr_min = np.min(foms)
            if curr_min < best_fom:
                best_fom = curr_min
                best_config = configs[np.argmin(foms)]
                print(f"  *** Breakthrough! New Best FOM: {best_fom:.4f} ***")
                
            print(f"  Added {num_new_samples} samples. Global Best FOM: {best_fom:.4f}")
        else:
            print("  No new unique configs found. Skipping simulation.")
            
        best_fom_history.append(best_fom)

    for i in range(1, size): comm.send(None, dest=i, tag=STOP_TAG)
    
    print("\n=== Finalizing ===")
    
    plot_fom_history(best_fom_history, output_folder)
    plot_optimization_trajectory(foms, PARAMS['INIT_DATASET_SIZE'], PARAMS['ADDING_NUM'], output_folder)
    
    if len(best_config) == Q_ROWS * Q_COLS:
        Q = np.array(best_config).reshape((Q_ROWS, Q_COLS))
        config_NxN = enforce_reflection_symmetry(Q)
    else:
        config_NxN = np.array(best_config).reshape((FULL_ROWS, FULL_COLS))
        
    final_config_NxN = config_NxN.T 
    np.save(os.path.join(output_folder, f"best_config_{FULL_ROWS}x{FULL_COLS}.npy"), final_config_NxN)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(final_config_NxN, cmap='gray_r', origin='lower', 
              extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2]) 
    
    ax.set_xticks(np.linspace(-MDM_LC/2, MDM_LC/2, FULL_COLS+1), minor=True)
    ax.set_yticks(np.linspace(-MDM_LC/2, MDM_LC/2, FULL_ROWS+1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0) 
    ax.set_axisbelow(False)
    
    ax.set_title(f"Best Binary Configuration ({FULL_ROWS}x{FULL_COLS})", fontsize=16)
    ax.set_xlabel("X ($\mu$m)", fontsize=14)
    ax.set_ylabel("Y ($\mu$m)", fontsize=14)
    plt.savefig(os.path.join(output_folder, f"best_config_{FULL_ROWS}x{FULL_COLS}.png"), dpi=300)
    plt.close()

    for i in range(1, size): comm.send(None, dest=i, tag=STOP_TAG)
        
    t_final_meep_start = time.time()
    final_res = perform_detailed_final_analysis(best_config, FULL_ROWS, FULL_COLS, output_folder)
    total_meep_time += time.time() - t_final_meep_start
    total_run_time = time.time() - start_total
    total_fm = sum(history['timing_metrics']['fm_train_time'])
    total_sa = sum(history['timing_metrics']['sa_time'])
    total_fdtd = total_meep_time
    other = total_run_time - (total_fm + total_sa + total_fdtd)
    plot_time_statistics_pie_chart(total_fm, total_sa, total_fdtd, other, output_folder)

    time_records = {
        "total_time_seconds": total_run_time,
        "total_meep_time_seconds": total_meep_time, 
        "avg_fm_train_time": float(np.mean(history['timing_metrics']['fm_train_time'])) if history['timing_metrics']['fm_train_time'] else 0,
        "avg_new_data_sim_time": float(np.mean(history['timing_metrics']['new_data_sim_time'])) if history['timing_metrics']['new_data_sim_time'] else 0,
        "total_fm_time": total_fm,
        "total_sa_time": total_sa,
        "total_fdtd_time": total_fdtd,
        "other_time": other
    }
    
    experiment_log = {
        "timestamp": timestamp,
        "hyperparameters": {
            "RESOLUTION": RESOLUTION,
            "MDM_LC": MDM_LC,
            "WG_LENGTH": WG_LENGTH,
            "WG_WIDTH": WG_WIDTH,
            "N_SIO2": N_SIO2,
            "N_SI": N_SI,
            "wl_cen": wl_cen,
            "FCEN": FCEN,
            "DF": DF,
            "NFREQ": NFREQ,
            "gaussian_kernel_size": KERNEL_SIZE,     
            "gaussian_kernel_sigma": KERNEL_SIGMA,   
            "tanh_beta": TANH_BETA,                  
            "tanh_eta": TANH_ETA,                    
            "smooth_threshold": SMOOTH_THRESHOLD,    
            "GRID_ROWS": FULL_ROWS,
            "GRID_COLS": FULL_COLS,
            "SAMPLER_TYPE": PARAMS['SAMPLER_TYPE'],
            "INIT_SIM_COUNT": PARAMS['INIT_DATASET_SIZE'],
            "ITERATIONS": PARAMS['ITERATIONS'],
            "SAMPLES_PER_ITER": PARAMS['ADDING_NUM'],
            "FM_EPOCHS": PARAMS['NUM_EPOCHS'],
            "FM_LR": PARAMS['LEARNING_RATE'],
            "FM_K": PARAMS['K_FACTOR'],
            "NUM_READS": PARAMS['NUM_READS'],
            "NUM_SWEEPS": NUM_SWEEPS,
        },
        "time_records": time_records,
        "results": {
            "best_fom": float(best_fom), 
            "best_latent_flat": best_config.tolist(), 
            "fom_evolution_history": best_fom_history,
            "all_evaluated_foms": foms.tolist(),
            "final_analysis": final_res
        }
    }
    
    for mode in final_res:
        for wl_key, vals in final_res[mode].items():
            experiment_log["results"][f"{mode} Through @ {wl_key}"] = f"{vals[0]:.4f} ({vals[1]:.2f} dB)"
    
    with open(os.path.join(output_folder, "final_result.json"), 'w') as f:
        json.dump(experiment_log, f, indent=4)
        
    print(f"Done. Results in {output_folder}")

if __name__ == "__main__":
    if size > 1 and rank != 0: worker_node()
    else: master_node()