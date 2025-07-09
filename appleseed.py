# Crack the global seed for Grey Hack using CUDA - input ip and network type
# Requires a valid CUDA installation (and ergo a compatible NVIDIA graphics card)

from numba import cuda
import numpy as np
import time

TipoRed = np.array([
    "Comisaria", "Universidades", "Supermercados", "FastFood", "Taller",
    "MobileShop", "Hospitales", "Bancos", "Particulares", "MailServices",
    "HackShop", "TiendaInformatica", "NetServices", "HardwareManufacturer",
    "Neurobox", "Tutorial", "CurrencyCreation"
])

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
    ip_ints = np.array([ip_to_int(ip) for _, ip in data], dtype=np.int32)
    expected_indices = np.array([idx for idx, _ in data], dtype=np.int32)

    # Allocate memory on GPU
    d_ip_ints = cuda.to_device(ip_ints)
    d_expected_indices = cuda.to_device(expected_indices)

    # Prepare results array
    result_size = 1024  # Allocate space for multiple possible seeds
    results = np.full(result_size, -1, dtype=np.int32)
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
    [7, "89.76.247.115"],
    [7, "98.102.48.136"],
    [7, "99.95.155.21"],
    [7, "99.92.97.100"],
    [7, "99.99.97.251"],
    [7, "99.96.0.37"],
    [7, "99.96.0.16"],
    [7, "99.96.1.0"],
    [7, "99.96.1.51"],
    [7, "99.96.1.85"],
    [7, "99.96.1.102"],
    [7, "99.96.1.68"],
    [7, "99.96.0.155"],
    [7, "99.96.0.206"],
    [7, "99.96.0.223"],
    [7, "99.96.2.116"],
    [9, "1.1.1.4"],
    [8, "1.1.1.5"],
    [16, "1.1.1.13"],
    [1, "1.1.1.15"],
]

# Run CUDA acceleration
now = time.time()
seed_result = run_cuda_find_matching_seeds(example_data)
print(f"Execution time: {time.time() - now:.2f} seconds")
print(f"Possible seed(s): {seed_result}")
