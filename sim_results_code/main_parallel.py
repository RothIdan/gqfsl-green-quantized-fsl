from energy_latency_models import E_U_k, E_C_k, E_T_k, T_sgd_k, Tx_time
from network_setup import get_network_params, get_channel_gains, build_params
from utils import expected_max_subset, convergence_bound, find_minimal_T
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
# from scipy.special import comb
import time
from datetime import datetime
from tqdm import tqdm
import os
import glob
import shutil
import argparse



def evaluate_scenario(I, s_point, settings_params, dc, ds, sys_params, temp_dir):
    # Unpack parameters from the system config
    N, G2, sigma2, L, mu, beta, Gamma, prob, q_max, b, max_split, epsilons, T_low, T_high = sys_params


    n_settings = len(settings_params)
    gamma = np.max([8*L/mu - 1, I])
    
    # If in FL case (s_point = last layer), loop exactly once (qs does not matter). Else, normal range.
    qs_list = [q_max] if s_point == max_split else range(4, q_max+1)

    results = []

    # 1. Precompute broadcast EC for each setting (only depends on s_point & K)
    broadcast_cache = np.zeros((n_settings, N + 1)) 
    for s_idx, eng_params in enumerate(settings_params):
        rates_inverse = []
        n_data, P_tx, B, N0_dBm = eng_params['update'][N].values() 
        N0 = 10**(N0_dBm/10) / 1000 
        
        for dev in range(N):
            h_k = eng_params['comm'][dev]['channel']
            rate_k = B * np.log2(1 + (P_tx * h_k) / (N0 * B))
            rates_inverse.append(1/rate_k)
        
        for K in range(1, N+1): 
            broadcast_cache[s_idx, K] = expected_max_subset(rates_inverse, K) * n_data * q_max * P_tx

    # 2. Precompute submodel upload EC & latency for each setting & for all clients
    upload_cache = {qu: [] for qu in range(4, q_max+1)}
    for qu in range(4, q_max+1):
        for eng_params in settings_params:
            e_com_arr = np.array([E_U_k(qu, eng_params['update'][dev]) for dev in range(N)])
            t_up_arr = np.array([Tx_time(qu, *eng_params['update'][dev].values()) for dev in range(N)])
            upload_cache[qu].append({'e_com': e_com_arr, 't_up': t_up_arr})

    # Main calculation loops
    for qc in range(4, q_max+1):
        for qs in qs_list:
            # 3. Precompute SGD EC and latency (only depends on qc & qs)
            sgd_cache = []
            for eng_params in settings_params:
                cmp_eng_arr, com_eng_arr = np.zeros(N), np.zeros(N)
                t_cmp_arr, t_com_arr = np.zeros(N), np.zeros(N)
                
                for dev in range(N):
                    cmp_eng_arr[dev] = E_C_k(qc, qs, eng_params['compute'][dev], eng_params['compute'][N]) 
                    com_eng_arr[dev] = E_T_k(qc, q_max, eng_params['comm'][dev], eng_params['comm'][N]) 
                    t_cmp_arr[dev], t_com_arr[dev] = T_sgd_k(qc, qs, eng_params['compute'][dev], eng_params['comm'][dev], eng_params['compute'][N], eng_params['comm'][N])

                sgd_cache.append({
                    'cmp_eng': cmp_eng_arr, 'com_eng': com_eng_arr,
                    't_cmp': t_cmp_arr, 't_com': t_com_arr
                })

            for qu in range(4, q_max+1):
                # Combine arrays for all settings BEFORE looping K
                settings_comp_eng, settings_comm_eng = np.zeros(n_settings), np.zeros(n_settings)
                setting_lat_arr, setting_comp_lat_arr, setting_comm_lat_arr = [], [], []

                for s_idx in range(n_settings):
                    sgd = sgd_cache[s_idx]
                    upl = upload_cache[qu][s_idx]
                    
                    e_cmp_arr = I * b * sgd['cmp_eng']
                    e_com_arr = I * b * sgd['com_eng'] + upl['e_com']
                    settings_comp_eng[s_idx] = np.sum(e_cmp_arr)
                    settings_comm_eng[s_idx] = np.sum(e_com_arr)

                    tau_comp = I * b * sgd['t_cmp']
                    tau_comm = I * b * sgd['t_com'] + upl['t_up']
                    setting_lat_arr.append(tau_comp + tau_comm)
                    setting_comp_lat_arr.append(tau_comp)
                    setting_comm_lat_arr.append(tau_comm)

                # Final loop over participating clients K
                for K in range(1, N+1):
                    final_eng, final_comp_eng, final_comm_eng = np.zeros(n_settings), np.zeros(n_settings), np.zeros(n_settings)
                    final_lat, final_comp_lat, final_comm_lat = np.zeros(n_settings), np.zeros(n_settings), np.zeros(n_settings)

                    for s_idx in range(n_settings):
                        e_comp = (K/N) * settings_comp_eng[s_idx]
                        e_comm = ((K/N) * settings_comm_eng[s_idx]) + broadcast_cache[s_idx, K]
                        final_comp_eng[s_idx] = e_comp
                        final_comm_eng[s_idx] = e_comm
                        final_eng[s_idx] = e_comp + e_comm 
                        
                        latencies = setting_lat_arr[s_idx] 
                        exp_lat = expected_max_subset(latencies, K) 
                        final_lat[s_idx] = exp_lat

                        ratio_comp = np.mean(setting_comp_lat_arr[s_idx]) / np.mean(latencies) 
                        final_comp_lat[s_idx] = exp_lat * ratio_comp
                        final_comm_lat[s_idx] = exp_lat * (1 - ratio_comp)

                    # Average across all settings
                    avg_energy = np.mean(final_eng)
                    avg_comp_eng = np.mean(final_comp_eng)
                    avg_comm_eng = np.mean(final_comm_eng)

                    avg_latency = np.mean(final_lat)
                    avg_comp_lat = np.mean(final_comp_lat)
                    avg_comm_lat = np.mean(final_comm_lat)

                    for eps in epsilons:
                    # Bisection Search of the range [T_low, T_high] using the specific 'eps'
                        T_min, alpha_star = find_minimal_T(T_low, T_high, convergence_bound, eps, G2, sigma2, L, mu, beta, Gamma, prob, gamma, dc, ds, qc, qs, qu, I, K, N)

                        if T_min is not None:
                            results.append({
                                "split": s_point, "qc": qc, "qs": qs if s_point != (n_layers-1) else 0, "qu": qu,  
                                "energy": T_min * avg_energy,
                                "energy_comp": T_min * avg_comp_eng,
                                "energy_comm": T_min * avg_comm_eng,
                                "latency": T_min * avg_latency,
                                "latency_comp": T_min * avg_comp_lat,
                                "latency_comm": T_min * avg_comm_lat,
                                "n_clients": K, "l_iter": I, "g_iter": T_min, "epsilon": eps, "alpha": alpha_star
                            })
                        
    # Automatic Checkpoint to disk for this chunk
    df = pd.DataFrame(results)
    chunk_filename = os.path.join(temp_dir, f"fsl_qnn_chunk_I{I}_split{s_point}.csv")
    df.to_csv(chunk_filename, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Wireless Quantized Federated Split Learning Simulation')
    parser.add_argument('-model', type=str, choices=['qfl', 'resnet18'], default='qfl', help='Architecture name for the QNN model to be used.')
    parser.add_argument('-name', type=str, help='filename tail for the output CSV file.')
    args = parser.parse_args()
    
    seed_value = 42
    rng = np.random.default_rng(seed=seed_value)

    # Load QNN architecture parameters
    network_layers_params, n_layers, x_in = get_network_params(args.model)
    max_split = len(network_layers_params) # Last layer split corresponds to FL case (no intermediate communication, qs does not matter)

    
    # System Constants
    b = 32
    N = 50 
    sigma2 = np.load("sigma.npy")**2
    L = 0.097 
    mu = 0.05
    Gamma = 0.6
    G2 = 0.25**2
    prob = np.ones(N)/N # Uniform sampling probability for all clients
    q_max = 32
    beta = 2/mu

    if args.model == 'qfl': # Base CNN on MNIST
        epsilons = [0.2, 0.18, 0.15, 0.13, 0.1, 0.08, 0.05, 0.03, 0.01]
    else: # ResNet18 on CIFAR-10
        epsilons = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    T_low, T_high = 1, 3000 # Bisection search range for the number of global rounds T
    
    # Pack parameters to pass to the workers
    sys_params = (N, G2, sigma2, L, mu, beta, Gamma, prob, q_max, b, max_split, epsilons, T_low, T_high)

    n_settings = 100 
    settings_config = []
    for _ in range(n_settings):
        channel_gains = get_channel_gains(rng, num_devices=N)
        clients_tiers = rng.integers(0, 3, size=N)
        # clients_tiers = np.ones(N, dtype=int) # For homogeneous tier (all mid-tier)
        settings_config.append({'channels': channel_gains, 'tiers': clients_tiers})

    # Prepare parameter dictionaries for all split points
    params = {} 
    for s_point in range(1, n_layers):
        Mc, Oc, dc = network_layers_params[s_point]['client'].values()
        Ms, Os, ds = network_layers_params[s_point]['server'].values()
        x_cut = network_layers_params[s_point]['act']

        params[s_point] = {'dc': dc, 'ds': ds, 'settings_params': []}

        for setting in settings_config:
            eng_params = build_params(Mc, Oc, dc, Ms, Os, ds, x_in, x_cut, setting['tiers'], setting['channels'])
            params[s_point]['settings_params'].append(eng_params)

    # Temporary directory for dropping finished chunks
    temp_dir = f"temp_chunks_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(temp_dir, exist_ok=True)


    # Create a list of all scenarios we want to run 
    tasks = []
    for I in [1, 2, 3]:
        for s_point in range(1, n_layers):
            expected_filename = os.path.join(temp_dir, f"fsl_qnn_chunk_I{I}_split{s_point}.csv")
            # Skip if already computed
            if os.path.exists(expected_filename):
                continue   
            tasks.append((I, s_point, params[s_point]['settings_params'], params[s_point]['dc'], params[s_point]['ds']))

    
    print(f"Parallel optimized search begins. Distributing {len(tasks)} tasks...")
    start_time = time.time()

    num_cores = max(1, multiprocessing.cpu_count() - 3)
    
    if len(tasks) > 0:
        with Parallel(n_jobs=num_cores, backend="loky") as parallel:
            parallel(
                delayed(evaluate_scenario)(I, s_point, s_params_dict, dc, ds, sys_params, temp_dir) 
                for I, s_point, s_params_dict, dc, ds in tqdm(tasks, desc="Overall Progress")
            )
            
        print("\nAll workers finished successfully!")

    else:
        print("\nAll chunks are already present! Skipping to merge.")

    # Merge chunks together
    all_chunks = glob.glob(os.path.join(temp_dir, "*.csv"))
    if all_chunks:
        print(f"Found {len(all_chunks)} chunk files. Merging...")
        combined_df = pd.concat((pd.read_csv(f) for f in all_chunks), ignore_index=True)
        combined_df.to_csv(f"fsl_qnn_optimization_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M')}_{args.name}.csv", index=False)
        
        # Cleanup temp chunks
        shutil.rmtree(temp_dir)
        print("Merge complete and temporary files cleaned up!")

    t = (time.time() - start_time)
    h = int(np.floor(t/3600))
    m = int(np.floor((t-h*3600)/60))
    s = int(np.floor(t-h*3600-m*60))
    print(f"Total Parallel Evaluation time: {h:0>2}:{m:0>2}:{s:0>2}")