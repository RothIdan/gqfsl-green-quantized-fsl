import numpy as np

#######################################
######    Latency Calculation    ######
#######################################

def Tx_time(q, h_gain, n_data, P_tx, B=10e6, N0_dBm=-174):
    """
    Calculates the wireless transmission latency over an OFDMA channel.

    Args:
        q (int): Quantization precision level in bits for the transmitted payload.
        h_gain (float): channel gain.
        n_data (int): Data payload size.
        P_tx (float): Transmission power in Watts.
        B (float): Allocated bandwidth in Hz.
        N0_dBm (float): Noise power spectral density in dBm/Hz.

    Returns:
        float: The transmission latency in seconds.
    """
    
    N0 = 10**(N0_dBm/10) / 1000 # Watt/Hz
    rate = B * np.log2(1 + (P_tx * h_gain) / (N0 * B))
    return n_data * q / rate


def T_fp(q, n_mac, p, f):
    '''
    Calculates the forward propagation latency.

    Args:
        q (int): The precision level for computaiton (32 bits).
        n_mac (int): The number of MACs for computation.
        p (int): The number of MAC units on the comptation platform.
        f (int): The computation platform clock frequency in Hz.
        
    Returns:
        float: The forward propagation latency.
    '''

    return n_mac / (p_eff(q, p) * f)


def T_sgd_k(qc, qs, device_compute_params, device_comm_params, server_compute_params, server_comm_params):
    """
    Calculates the total local computation and intermediate communication latency for one SGD step.
    This function accounts for the forward and backward passes split between the client and server.

    Args:
        qc (int): Client-side submodel precision level in bits.
        qs (int): Server-side submodel precision level in bits.
        device_compute_params (dict): Hardware specifications for the client device.
        device_comm_params (dict): Uplink channel and transmission parameters.
        server_compute_params (dict): Hardware specifications for the allocated server instance.
        server_comm_params (dict): Downlink channel and transmission parameters.

    Returns:
        tuple: A tuple containing total computation latency (float) and total communication latency (float) in seconds.
    """
    q_max = device_compute_params['q_max'] # Maximal precision level (32 bits)

    client_fp = T_fp(qc, device_compute_params['Mc'], device_compute_params['pc'], device_compute_params['fc']) 
    server_fp = T_fp(qs, server_compute_params['Ms'], server_compute_params['p'], server_compute_params['f'])

    # BP is done on full precision + assumed to have 2 * MACs compared to FP
    client_bp = 2 * T_fp(q_max, device_compute_params['Mc'], device_compute_params['pc'], device_compute_params['fc']) 
    server_bp = 2 * T_fp(q_max, server_compute_params['Ms'], server_compute_params['p'], server_compute_params['f'])
    
    activation_tx = Tx_time(qc, *device_comm_params.values())
    gradient_tx = Tx_time(q_max, device_comm_params["channel"], *server_comm_params.values()) 

    comp_time = client_fp + server_fp + server_bp + client_bp
    comm_time = activation_tx + gradient_tx

    return comp_time, comm_time



##################################################
######    Energy Consumption Calculation    ######
##################################################

def E_MAC(q, q_max=32, A=3.7e-12, alpha=1.25):
    """
    Calculates the precision-dependent energy consumption of a single MAC operation.

    Args:
        q (int): The current quantization precision level in bits.
        q_max (int): The maximum full-precision level in bits (default is 32).
        A (float): Hardware-specific scaling constant in Joules.
        alpha (float): Non-linear scaling factor for precision.

    Returns:
        float: Energy consumed by a single MAC operation in Joules.
    """
    return A * (q / q_max) ** alpha

# Read/write energy for local buffer
def E_l(q, A_l=1): 
    return A_l * E_MAC(q)

# Read/write energy for main SRAM
def E_m(q, A_m=2):
    return A_m * E_MAC(q)

# Read/write energy for DRAM
def E_d(q, A_d):
    return A_d * E_MAC(q)

# The effective number of reduced precision MAC units
def p_eff(q,p,q_max=32):
    return p * (q_max / q)


def E_forward(q, n_mac, n_act, n_weight, p, S, q_max=32, A_d=150, x_in=784, is_device=True):
    """
    Calculates the forward propagation energy consumption for a single data sample.
    This models the total energy across MAC units, local buffer, main SRAM, and off-chip DRAM.

    Args:
        q (int): QNN precision level in bits.
        n_mac (int): Number of MAC operations for the submodel.
        n_act (int): Total number of activation outputs for the submodel.
        n_weight (int): Total number of submodel weights.
        p (int): Baseline number of physical MAC units on the platform.
        S (int): Main SRAM size capacity in bits.
        q_max (int): Maximum QNN precision level in bits.
        A_d (int): Read/write energy consumption factor for DRAM relative to a MAC operation.
        x_in (int): Input dimension size in bits.
        is_device (bool): Flag indicating if the computation is on the client device.

    Returns:
        float: Total forward propagation energy consumption in Joules.
    """
    
    EC = E_MAC(q) * n_mac + 3 * n_act * E_MAC(q_max)
    EW = E_m(q) * n_weight + E_l(q) * n_mac / np.sqrt(p_eff(q, p))
    EA = 2 * E_m(q) * n_act + E_l(q) * n_mac / np.sqrt(p_eff(q, p))
    if is_device:
        EDRAM = E_d(q_max, A_d) * x_in + E_d(q, A_d) * max(n_weight * q - S/2, 0) + 2 * E_d(q, A_d) * max(n_act * q - S/2, 0)
    else: # server
        EDRAM = E_d(q, A_d) * x_in + E_d(q, A_d) * max(n_weight * q - S/2, 0) + 2 * E_d(q, A_d) * max(n_act * q - S/2, 0) # cut-layer activations are in q precision level

    return EC + EW + EA + EDRAM


# Backpropagation energy consumption for a single data sample
# The number of MACs is assumed to be 2x the number of MACs in the forward pass
def E_back(n_mac, n_act, n_weight, p, S, q_max, A_d):
   return 2 * n_mac * E_MAC(q_max) + 2 * E_m(q_max) * n_act + E_m(q_max) * n_weight + \
          2 * E_l(q_max) * 2*n_mac / np.sqrt(p_eff(q_max, p)) + E_d(q_max, A_d) * max(n_weight * q_max - S/2, 0) + \
          2 * E_d(q_max, A_d) * max(n_act * q_max - S/2, 0)



# Total computation energy consumption for one SGD step over one data sample
def E_C_k(qc, qs, device_params, server_params): 
    device_fp = E_forward(qc, *list(device_params.values())[1:], is_device=True)
    server_fp = E_forward(qs, *list(server_params.values())[1:], is_device=False)
    device_bp = E_back(*list(device_params.values())[1:-1]) # func. gets as input all parameters except the last
    server_bp = E_back(*list(server_params.values())[1:-1])
    
    return device_fp + server_fp + device_bp + server_bp

# Intermidiate transmission energy consumption to support local SGD step
def E_T_k(qc, q_max, device_params, server_params): 
    activation_tx = Tx_time(qc, *device_params.values()) * device_params['P_ul']
    gradient_tx = Tx_time(q_max, device_params["channel"], *server_params.values()) * server_params['P_dl']
    
    return activation_tx + gradient_tx

# Model upload energy cosumption for aggregation at the server
def E_U_k(q, params):
    return Tx_time(q, *params.values()) * params['P_ul']

# # Broadcasting energy consuumption for updated model distribution to all clients
# def E_B(q, h_avg, params):
#     return Tx_time(q, h_avg, *params.values()) * params['P_dl']


