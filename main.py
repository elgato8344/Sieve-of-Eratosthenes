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
