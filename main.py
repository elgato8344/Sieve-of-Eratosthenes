import time

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
