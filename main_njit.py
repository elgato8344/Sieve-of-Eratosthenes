from numba import njit
import numpy as np

@njit(cache=True)
def is_prime_v3(number: int):
    if number < 2:
        return np.empty(0, dtype=np.int64)
    size = (number - 1) // 2 + 1
    numbers = np.ones(size, dtype=np.uint8)
    numbers[0] = 0
    limit = int(number ** 0.5)
    for i in range(1, (limit - 1) // 2 + 1):
        if numbers[i]:
            p = 2 * i + 1
            start_idx = (p * p - 1) // 2
            for idx in range(start_idx, size, p):
                numbers[idx] = 0
    odd_indices = np.nonzero(numbers)[0]
    primes = 2 * odd_indices + 1
    result = np.empty(len(primes) + 1, dtype=np.int64)
    result[0] = 2
    result[1:] = primes
    return result

def sieve_numba(number: int):
    start = time.perf_counter()
    primes = is_prime_v3(number)
    e_time = time.perf_counter() - start

    print(f"It took {e_time:.10f} seconds to complete up to {number}.(numba machine code compiled)")
    print(f"{len(primes)} primes found")
    return primes

sieve_numba(1000000000)
sieve_numba(1000000000)
sieve_numba(1000000000)
sieve_numba(1000000000)
sieve_numba(1000000000)
sieve_numba(1000000000)
