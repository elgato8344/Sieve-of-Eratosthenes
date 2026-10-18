import math
import time
from concurrent.futures import ProcessPoolExecutor


def _sieve_single_segment(low_val: int, high_val: int, base_primes: list):
    """Worker function executed by each individual CPU core."""
    current_segment_size = (high_val - low_val) // 2 + 1
    segment = bytearray(b"\x01") * current_segment_size

    for p in base_primes:
        if p * p > high_val:
            break

        # Find the first odd multiple of p that is >= low_val
        first_multiple = ((low_val + p - 1) // p) * p
        if first_multiple % 2 == 0:
            first_multiple += p

        # Map to the local relative index inside this worker's segment window
        start_idx = (first_multiple - low_val) // 2
        step = p

        # Blit across the bytearray at native C-speeds inside the process
        if start_idx < current_segment_size:
            num_multiples = (current_segment_size - 1 - start_idx) // step + 1
            segment[start_idx : current_segment_size : step] = (
                b"\x00" * num_multiples
            )

    # Return the verified primes found in this process chunk
    return [
        low_val + 2 * i for i, is_prime_val in enumerate(segment) if is_prime_val
    ]


def get_primes_parallel(number: int):
    start = time.perf_counter()

    if number < 2:
        return []

    # step 1: Main process generates sequential Base Primes up to sqrt(N)
    limit = int(math.isqrt(number))
    base_size = (limit - 1) // 2 + 1 if limit >= 1 else 1
    base_sieve = bytearray(b"\x01") * base_size
    base_sieve[0] = 0

    for i in range(1, (int(limit**0.5) - 1) // 2 + 1):
        if base_sieve[i]:
            p = 2 * i + 1
            start_idx = (p * p - 1) // 2
            num_multiples = (base_size - 1 - start_idx) // p + 1
            base_sieve[start_idx : base_size : p] = b"\x00" * num_multiples

    base_primes = [
        2 * i + 1 for i, is_prime_val in enumerate(base_sieve) if is_prime_val
    ]

    primes = [2] if number >= 2 else []
    primes.extend(base_primes)

    # step 2: Plan the Parallel Segments
    SEGMENT_SIZE = 4_000_000

    low_val = limit + 1 if (limit + 1) % 2 != 0 else limit + 2
    if low_val < 3:
        low_val = 3

    tasks = []
    while low_val <= number:
        high_val = min(low_val + 2 * SEGMENT_SIZE - 2, number)
        tasks.append((low_val, high_val))
        low_val = high_val + 2

    # step 3: Farm out chunks using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_sieve_single_segment, low, high, base_primes)
            for low, high in tasks
        ]

        for future in futures:
            primes.extend(future.result())

    stop = time.perf_counter()
    e_time = stop - start

    print(f"Parallel Sieve took {e_time:.4f} seconds up to {number}. (bytearray with multiple threads)")
    print(f"{len(primes)} primes found.")

    return primes


# --- REQUIRED FOR MULTIPROCESSING IN PYTHON ---
if __name__ == "__main__":
    get_primes_parallel(1_000_000_000)
    get_primes_parallel(1_000_000_000)
    get_primes_parallel(1_000_000_000)
