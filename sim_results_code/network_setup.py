import numpy as np



def get_channel_gains(rng, num_devices, cell_length=500, alpha=4, K=1, d0=1):
    """
    Calculates the channel gains for a given number of clients based on path loss.
    Clients are uniformly distributed within a specified cell area around a central base station.

    Args:
        rng (Generator): NumPy random number generator instance.
        num_devices (int): Number of clients in the network.
        cell_length (int): The maximum dimension of the square simulation area in meters.
        alpha (float): Path loss exponent.
        K (float): Path loss constant factor.
        d0 (float): Reference distance in meters.

    Returns:
        ndarray: An array containing the channel gain for each client.
    """

    # Generate Client Locations (ensures min_dist >= sqrt(2)) assuming BS is at the center
    x = 1 + (cell_length/2 - 1) * rng.random(num_devices)
    y = 1 + (cell_length/2 - 1) * rng.random(num_devices)
    
    # Calculate Distances
    distances = np.sqrt(x**2 + y**2)

    # Calculate path loss
    pl = K * (d0 / distances)**alpha

    channel_gains = pl

    return channel_gains




#####################################
######    System Parameters    ######
#####################################

device_compute_defaults = {
    "q_max": 32,
    "A_d": 150
}

server_compute_defaults = {
    "q_max": 32,
    "A_d": 150
}

device_comm_defaults = {
    # "h_ul": 100**-4,
    "P_ul": 0.1,
    "B_ul": 4e6,
    "N0_dbm": -174,
}

server_comm_defaults = {
    "P_dl": 0.1,
    "B_dl": 4e6,
    "N0_dbm": -174,
}

# Optional: different transmission related parameters might be used for model update
device_update_defaults = {
    "P_ul": 0.1,
    "B_ul": 4e6,
    "N0_dbm": -174,
}

server_update_defaults = {
    "P_dl": 5,
    "B_dl": 20e6,
    "N0_dbm": -174,
}


# === Function to build params dynamically ===
def build_params(Mc, Oc, dc, Ms, Os, ds, x_in, x_cut, client_tiers, channel_gains):
    """
    Builds the system parameter dictionaries for all clients and the edge server.
    This maps hardware tier specifications (MACs, SRAM, clock frequency) to specific clients.

    Args:
        Mc (int): Number of MAC operations in the client submodel.
        Oc (int): Number of activations in the client submodel.
        dc (int): Number of weights in the client submodel.
        Ms (int): Number of MAC operations in the server submodel.
        Os (int): Number of activations in the server submodel.
        ds (int): Number of weights in the server submodel.
        x_in (int): Input feature dimension size.
        x_cut (int): Intermediate activation payload size at the cut layer.
        client_tiers (array-like): Array mapping each client to a specific hardware tier index.
        channel_gains (array-like): Array containing the channel gain for each client.

    Returns:
        dict: A nested dictionary containing 'compute', 'comm', and 'update' parameters.
    """
    params = {"compute": {}, "comm": {}, "update": {}}

    num_devices = len(channel_gains)

    # Clients hardware tiers (compute + memory scale together + frequency)
    hardware_tiers = [
        (16,  0.5e6 * 8, 200e6), # Low Tier:  16 MACs, 0.5 MB SRAM, 200 MHz frequency 
        (64,  2.0e6 * 8, 500e6), # Mid Tier:  64 MACs, 2.0 MB SRAM, 500 MHz frequency 
        (256, 4.0e6 * 8, 1.2e9)  # High Tier: 256 MACs, 4.0 MB SRAM, 1.2 GHz frequency 
    ]

    # Devices
    for dev in range(num_devices):
        tier_idx = client_tiers[dev]
        pc_val, Sc_val, fc_val = hardware_tiers[tier_idx]
        # pc_val, Sc_val, fc_val = hardware_tiers[1]
        
        params["compute"][dev] = {"fc": fc_val, "Mc": Mc, "Oc": Oc, "dc": dc, "pc": pc_val, "Sc": Sc_val, **device_compute_defaults, "x_in": x_in}
        params["comm"][dev] = {"channel": channel_gains[dev], "x_cut": x_cut, **device_comm_defaults}
        params["update"][dev] = {"channel": channel_gains[dev], "dc": dc, **device_update_defaults}

    # Server compute params
    p, S, f = (256, 4.0e6 * 8, 1.2e9)

    params["compute"][num_devices] = {"f": f, "Ms": Ms, "Os": Os, "ds": ds, "p": p, "S": S, **server_compute_defaults, "x_in": x_cut}
    params["comm"][num_devices] = {"x_cut": x_cut, **server_comm_defaults}
    params["update"][num_devices] = {"dc": dc, **server_update_defaults}
    
    return params



def get_network_params(name='qfl'):
    """
    Returns the NN architecture per cut-layer parameters.

    Args:
        name (str): The name of the neural network architecture. 
                    Options are 'qfl' for a base CNN on MNIST or 'resnet18' for ResNet18 on CIFAR10.

    Returns:
        tuple: A tuple containing:
            - params (dict): Nested dictionary with split-specific layer metrics, structured as:
                - Mc (int): Number of MAC operations in the client submodel.
                - Oc (int): Number of activations in the client submodel.
                - dc (int): Number of weights in the client submodel.
                - Ms (int): Number of MAC operations in the server submodel.
                - Os (int): Number of activations in the server submodel.
                - ds (int): Number of weights in the server submodel.
                - act / x_cut (int): Intermediate activation payload size at the cut layer.
            - n_layers (int): The total number of candidate split points plus one.
            - x_in (int): The input feature dimension size.
    """

    # Base CNN on MNIST (1x28x28)
    if name == 'qfl':
        x_in = 1*28*28
        params = {
            1: {'client' : {'Mc' : 778752,   'Oc' : 73728, 'dc' : 1152+128*3},
                'server' : {'Ms' : 39119656, 'Os' : 13790, 'ds' : 405926},
                'act'  : 73728},
            2: {'client' : {'Mc' : 36463104, 'Oc' : 80128, 'dc' : 75456},
                'server' : {'Ms' : 3435304,  'Os' : 7390,  'ds' : 332006},
                'act'  : 6400},
            3: {'client' : {'Mc' : 38822400, 'Oc' : 84224, 'dc' : 112512},
                'server' : {'Ms' : 1076008,  'Os' : 3294,  'ds' : 294950},
                'act'  : 4096},
            4: {'client' : {'Mc' : 39485952, 'Oc' : 85376, 'dc' : 131040},
                'server' : {'Ms' : 412456,   'Os' : 2142,  'ds' : 276422},
                'act'  : 1152},
            5: {'client' : {'Mc' : 39633408, 'Oc' : 85408, 'dc' : 140352},
                'server' : {'Ms' : 265000,   'Os' : 2110,  'ds' : 267110},
                'act'  : 32},
            6: {'client' : {'Mc' : 39697408, 'Oc' : 87408, 'dc' : 206352},
                'server' : {'Ms' : 201000,   'Os' : 110,   'ds' : 201110},
                'act'  : 2000},
            7: {'client' : {'Mc' : 39897408, 'Oc' : 87508, 'dc' : 406452},
                'server' : {'Ms' : 1000,     'Os' : 10,    'ds' : 1010},
                'act'  : 100},
            8: {'client' : {'Mc' : 39898408, 'Oc' : 87518, 'dc' : 407462},
                'server' : {'Ms' :0,         'Os' : 0,     'ds' : 0},
                'act'  : 0}
        }

    # ResNet18 (with first conv layer modified) on CIFAR-10 (3x32x32)
    elif name == 'resnet18':
        x_in = 3*32*32
        params = {
            1: {'client': {'Mc': 1769472, 'Oc': 65536, 'dc': 1856},
                'server': {'Ms': 553661952, 'Os': 245770, 'ds': 11172106},
                'act': 65536},
            2: {'client': {'Mc': 77266944, 'Oc': 131072, 'dc': 75840},
                'server': {'Ms': 478164480, 'Os': 180234, 'ds': 11098122},
                'act': 65536},
            3: {'client': {'Mc': 152764416, 'Oc': 196608, 'dc': 149824},
                'server': {'Ms': 402667008, 'Os': 114698, 'ds': 11024138},
                'act': 65536},
            4: {'client': {'Mc': 211484672, 'Oc': 229376, 'dc': 379968},
                'server': {'Ms': 343946752, 'Os': 81930, 'ds': 10793994},
                'act': 32768},
            5: {'client': {'Mc': 286982144, 'Oc': 262144, 'dc': 675392},
                'server': {'Ms': 268449280, 'Os': 49162, 'ds': 10498570},
                'act': 32768},
            6: {'client': {'Mc': 345702400, 'Oc': 278528, 'dc': 1594432},
                'server': {'Ms': 209729024, 'Os': 32778, 'ds': 9579530},
                'act': 16384},
            7: {'client': {'Mc': 421199872, 'Oc': 294912, 'dc': 2775104},
                'server': {'Ms': 134231552, 'Os': 16394, 'ds': 8398858},
                'act': 16384},
            8: {'client': {'Mc': 479920128, 'Oc': 303104, 'dc': 6448192},
                'server': {'Ms': 75511296, 'Os': 8202, 'ds': 4725770},
                'act': 8192},
            9: {'client': {'Mc': 555417600, 'Oc': 311296, 'dc': 11168832},
                'server': {'Ms': 13824, 'Os': 10, 'ds': 5130},
                'act': 8192},
            10: {'client': {'Mc': 555431424, 'Oc': 311306, 'dc': 11173962},
                'server': {'Ms': 0, 'Os': 0, 'ds': 0},
                'act': 0}
        }

    return params, len(params)+1, x_in