#  The Sieve of Eratosthenes
# • The Math: An ancient algorithm for finding all prime numbers up to a given
# limit N by crossing off multiples of each prime starting from 2.
# • The Challenge: Implement the sieve using a Python list or bytearray, and
# time how fast it can find all primes up to 1,000,000.
import time

def find_prime(number: int):
    start = time.perf_counter()
    numbers = {x: True for x in range(2, number + 1)}
    for p in range(2, int(number**0.5) +1):
        if numbers[p]:
            for multiples in range(p * p, number + 1, p):
                numbers[multiples] = False
    stop = time.perf_counter()
    e_time = stop - start
    minutes, seconds = divmod(e_time, 60)
    primes = [(key, value) for key, value in numbers.items() if value is True]
    for key, value in primes:
        print(f"{key} : {value}")
    print(f"{len(primes)} primes found")
    print(f"It took {seconds:.4f} seconds to complete up to {number}.")

find_prime(1000000)

# For the record:
# 78498 primes found
# It took 0 minutes and 0.1086 seconds to complete up to 1000000.

def is_prime(number: int):
    """This version uses lists entirely instead of dictionary from my initial version.
    The difference in time it takes to go through a large dictionary vs a list
    is kind of wild.."""
    start = time.perf_counter()
    numbers = [True] * (number + 1)
    numbers[0] = numbers[1] = False
    for p in range(2, int(number**0.5) + 1):
        if numbers[p]:
            for multiples in range(p * p, number + 1, p):
                numbers[multiples] = False
    stop = time.perf_counter()
    e_time = stop - start
    minutes, seconds = divmod(e_time, 60)
    primes = [index for index, is_prime in enumerate(numbers) if is_prime]

    print(f"It took {seconds:.4f} seconds to complete up to {number}.")
    print(f"{len(primes)} primes found")
    return primes

primes_list = is_prime(1000000)
# This version is much faster. Since lists store elements in contiguous memory slots,
# Python can look up or change a value at any index nearly instantly,
# hence the faster work time.
# It took 0.0319 seconds to complete up to 1000000. ## !!! Changing to lists was a 240% increase in speed!
# 78498 primes found

# After thinking about this a bit and doing some research on lists and contiguous
# memory bits, I learned a bit about bytearrays and I'm going to try to make
# a new version

def is_prime_v2(number: int):
    start = time.perf_counter()
    if number < 2:
        return []
    if number == 2:
        return [2]
    size = (number -1) // 2 + 1
    numbers = bytearray(b"\x01") * size
    numbers[0] = 0
    limit = int(number ** 0.5)

    for i in range(1, (limit -1) // 2 + 1):
        if numbers[i]:
            p = 2 * i + 1
            start_idx = (p * p - 1) // 2
            step = p
            num_multiples = (size - 1 - start_idx) // step + 1
            numbers[start_idx : size : step] = b"\x00" * num_multiples
    stop = time.perf_counter()
    e_time = stop - start

    primes = [2] + [2 * i + 1 for i, is_prime_v2 in enumerate(numbers) if is_prime_v2]
    print(f"It took {e_time:.6f} seconds to complete up to {number}.(original bytearray)")
    print(f"{len(primes)} primes found")
    return primes

is_prime_v2(1000000000)
# # It took 0.000646 seconds to complete up to 1000000.
# # 78498 primes found
# # Okay, bytearrays are wayyy faster.. 4,838% faster..
# # FOUR THOUSAND PERCENT
#
# # Okay... wait. After doing some more research on how bytearrays work and interact with memory, I found a module called numpy and numba. Watch this:
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

# numpy with numba in MOST cases should be outperforming bytearray and python entirely
# What I found testing this was my machines maximum cache speed limit and found the
# bytearrays will pull ahead always because Python doesn't step through the array
# one element at a time like a normal loop. Under the hood, Python's C source code treats
# bytearray as a contiguous block of raw memory bytes. It passes this stride straight
# to the CPU as a highly optimized vector operation. It is running at the absolute
# maximum hardware speed my RAM can sustain every time I run it.
# In the numba version, even though it is compiled to machine code, the inner loop
# still explicitly increments an index and updates memory locations one by one. At a
# scale of 1 billion, my CPU cache is constantly clearing and reloading data from my
# main RAM (cache thrashing). Because Numba is evaluating these memory jumps one
# instruction at a time, it cannot outpace the low-level, block-level memory blitting
# that Python's native bytearray slice engine is doing.
# I expected the final Python list comprehension to bottleneck the bytearray version
# due to Python's object creation overhead. At 1 billion range, there are 50,847,534
# prime numbers.
# A python list containing ~50 million integers should take up roughly 400 MB of RAM.
# If I have fast RAM and a good CPU, Python can append 50 million elements into a
# list in less than a second, meaning it never slows down enough for Numba's
# fixed-size NumPy array allocation to pull ahead.
# I believe this could be the, or close to the fastest pure-Python Sieve of Eratosthenes
# possible without segmenting the target number. IT somehow managed to beat a
# dedicated machine-code compiler (numba) without needing any heavy external
# dependencies.


"""This is my last and final attempt to make this as fast as humanly possible."""
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

# It took 3.467792 seconds to complete up to 1000000000.(original bytearray)
# It took 4.0612324000 seconds to complete up to 1000000000.(numba machine code compiled)
# Parallel Sieve took 3.7015 seconds up to 1000000000. (bytearray with multiple threads)
# This is WILD lol
# single-threaded bytearray still beating out multi-core processing AND dedicated
# machine-code compiler
# I did a bit more research on this and believe I have encountered a fascinating
# architectural ceiling in Python called the Inter-Process Communication (IPS) Bottleneck.
# I think I'm done with this one for a while but, I loved learning about this.
