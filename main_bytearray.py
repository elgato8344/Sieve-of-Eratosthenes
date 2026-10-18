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
