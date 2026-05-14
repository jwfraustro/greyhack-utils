# Crack the global seed for Grey Hack using CUDA - input ip and network type
# Requires a valid CUDA installation (and ergo a compatible NVIDIA graphics card)

import time

import numpy as np
from numba import cuda

TipoRed = np.array(
    [
        "Comisaria",
        "Universidades",
        "Supermercados",
        "FastFood",
        "Taller",
        "MobileShop",
        "Hospitales",
        "Bancos",
        "Particulares",
        "MailServices",
        "HackShop",
        "TiendaInformatica",
        "NetServices",
        "HardwareManufacturer",
        "Neurobox",
        "CurrencyCreation",
    ]
)


# Convert IP to integer
def ip_to_int(ip):
    array = list(map(int, ip.split(".")))
    return (array[0] << 24) | (array[1] << 16) | (array[2] << 8) | array[3]


# CUDA Kernel: Find matching seeds in parallel
@cuda.jit
def find_matching_seeds_cuda(ip_ints, expected_indices, tipo_red_length, results):
    seed = cuda.grid(1)  # Get global thread ID

    if seed >= 0x7FFFFFFF:  # Prevent out-of-bounds access
        return

    # Check if the seed satisfies all conditions
    for i in range(ip_ints.shape[0]):
        computed_index = ((ip_ints[i] ^ seed) & 0x7FFFFFFF) % tipo_red_length
        if computed_index != expected_indices[i]:
            return  # Mismatch found, discard seed

    # If a valid seed is found, store it
    results[seed % results.size] = seed


# Main function to execute CUDA kernel
def run_cuda_find_matching_seeds(data):
    tipo_red_length = len(TipoRed)

    # Convert IPs to integers and extract expected indices
    ip_ints = np.array([ip_to_int(ip) for _, ip in data], dtype=np.int64)
    expected_indices = np.array([idx for idx, _ in data], dtype=np.int64)

    # Allocate memory on GPU
    d_ip_ints = cuda.to_device(ip_ints)
    d_expected_indices = cuda.to_device(expected_indices)

    # Prepare results array
    result_size = 1024  # Allocate space for multiple possible seeds
    results = np.full(result_size, -1, dtype=np.int64)
    d_results = cuda.to_device(results)

    # Define CUDA thread layout
    threads_per_block = 256
    blocks_per_grid = (0x7FFFFFFF // threads_per_block) + 1

    # Launch GPU Kernel
    find_matching_seeds_cuda[blocks_per_grid, threads_per_block](
        d_ip_ints, d_expected_indices, tipo_red_length, d_results
    )

    # Copy results back to CPU
    results = d_results.copy_to_host()

    # Filter valid seeds
    valid_seeds = set(results[results != -1])
    return valid_seeds


# A mapping of the TipoRed (website type) to IP address
example_data = [
    [7, "99.71.91.182"],
    [7, "99.71.254.118"],
    [7, "99.7.39.118"],
    [7, "99.68.173.118"],
    [7, "99.97.252.166"],
    [9, "99.71.149.216"],
    [9, "99.70.83.72"],
    [0, "99.8.188.49"],
    [0, "99.101.209.161"],
]

# Run CUDA acceleration
now = time.time()
seed_result = run_cuda_find_matching_seeds(example_data)
print(f"Execution time: {time.time() - now:.2f} seconds")
print(f"Possible seed(s): {len(seed_result)}")
print(f"Possible seed(s): {seed_result}")
