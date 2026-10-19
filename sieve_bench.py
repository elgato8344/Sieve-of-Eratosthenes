"""
Sieve of Eratosthenes benchmark: bytearray vs Numba vs multiprocessing.

Every version is split into phases so the comparison is fair:
  * sieve only   - build the odd-only flag array (or count primes)
  * extract      - turn flags into an actual collection of primes
  * full         - sieve + extract

Usage:
    python sieve_bench.py                    # N = 1_000_000_000, 5 runs
    python sieve_bench.py 100000000 --runs 3
    python sieve_bench.py --include-original # also time the original list comprehension
"""
import argparse
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import compress

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

# Known values of pi(N), used to verify every version.
KNOWN = {
    10**6: 78_498,
    10**7: 664_579,
    10**8: 5_761_455,
    10**9: 50_847_534,
}

# Flags (= odd numbers) per segment in the parallel version.
# 256 Ki flags covers 512K numbers and fits comfortably in L2 cache.
SEGMENT_FLAGS = 256 * 1024


# ---------------------------------------------------------------------------
# Version 1: pure Python bytearray (single process)
# Index i in the flag array represents the odd number 2*i + 1.
# ---------------------------------------------------------------------------
def sieve_bytearray(number: int) -> bytearray:
    """Phase 1: build the odd-only flag array."""
    size = (number - 1) // 2 + 1
    flags = bytearray(b"\x01") * size
    flags[0] = 0  # 1 is not prime
    limit = math.isqrt(number)
    for i in range(1, (limit - 1) // 2 + 1):
        if flags[i]:
            p = 2 * i + 1
            start = (p * p - 1) // 2
            flags[start:size:p] = bytes(len(range(start, size, p)))
    return flags


def extract_bytearray(flags: bytearray, number: int) -> list:
    """Phase 2: flags -> list of ints. itertools.compress keeps the loop in C."""
    primes = [2]
    primes.extend(compress(range(1, 2 * len(flags), 2), flags))
    return primes


def extract_bytearray_original(flags: bytearray, number: int) -> list:
    """The original list comprehension, kept for comparison (much slower)."""
    return [2] + [2 * i + 1 for i, f in enumerate(flags) if f]


# ---------------------------------------------------------------------------
# Version 2: Numba njit (single thread)
# ---------------------------------------------------------------------------
if HAVE_NUMBA:

    @njit(cache=True)
    def sieve_numba(number):
        """Phase 1: build the odd-only flag array."""
        size = (number - 1) // 2 + 1
        flags = np.ones(size, dtype=np.uint8)
        flags[0] = 0
        limit = int(number ** 0.5)
        while limit * limit > number:
            limit -= 1
        while (limit + 1) * (limit + 1) <= number:
            limit += 1
        for i in range(1, (limit - 1) // 2 + 1):
            if flags[i]:
                p = 2 * i + 1
                for idx in range((p * p - 1) // 2, size, p):
                    flags[idx] = 0
        return flags

    @njit(cache=True)
    def extract_numba(flags):
        """Phase 2: count, then fill. One output array, no temporaries."""
        count = 0
        for i in range(flags.size):
            count += flags[i]
        out = np.empty(count + 1, dtype=np.int64)
        out[0] = 2
        j = 1
        for i in range(flags.size):
            if flags[i]:
                out[j] = 2 * i + 1
                j += 1
        return out


# ---------------------------------------------------------------------------
# Version 3: segmented sieve across processes (multiprocessing, not threads)
# ---------------------------------------------------------------------------
def _segment_flags(low: int, high: int, base_primes: list) -> bytearray:
    """Sieve the odd numbers low, low+2, ..., <= high. `low` must be odd."""
    size = (high - low) // 2 + 1
    seg = bytearray(b"\x01") * size
    for p in base_primes:
        if p * p > high:
            break
        first = ((low + p - 1) // p) * p
        if first % 2 == 0:
            first += p
        start = (first - low) // 2
        if start < size:
            seg[start:size:p] = bytes(len(range(start, size, p)))
    return seg


def _worker_count(args):
    low, high, base_primes = args
    return _segment_flags(low, high, base_primes).count(1)


def _worker_primes(args):
    low, high, base_primes = args
    seg = _segment_flags(low, high, base_primes)
    idx = np.flatnonzero(np.frombuffer(seg, dtype=np.uint8))
    return idx.astype(np.int64) * 2 + low


def sieve_parallel(number: int, mode: str, segment_flags: int = SEGMENT_FLAGS):
    """
    mode="count":  return the number of primes (sieve only, no extraction)
    mode="primes": return a sorted np.int64 array of all primes
    Timing of this function includes process-pool startup.
    """
    if number < 2:
        return 0 if mode == "count" else np.empty(0, dtype=np.int64)

    limit = math.isqrt(number)
    base_flags = sieve_bytearray(max(limit, 1))
    base_primes = list(compress(range(1, 2 * len(base_flags), 2), base_flags))

    tasks = []
    low = limit + 1 if (limit + 1) % 2 else limit + 2
    low = max(low, 3)
    while low <= number:
        high = min(low + 2 * (segment_flags - 1), number)
        tasks.append((low, high, base_primes))
        low = high + 2

    worker = _worker_count if mode == "count" else _worker_primes
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(worker, tasks, chunksize=1))

    if mode == "count":
        return 1 + len(base_primes) + sum(results)
    head = np.array([2] + base_primes, dtype=np.int64)
    return np.concatenate([head] + results)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t


def check(label, count, number):
    expected = KNOWN.get(number)
    if expected is not None and count != expected:
        raise SystemExit(f"WRONG RESULT in {label}: got {count}, expected {expected}")
    return count


def run_all(number, runs, include_original):
    rows = {}  # label -> {"sieve": [...], "extract": [...], "full": [...]}
    counts = {}

    def add(label, sieve=None, extract=None, full=None):
        r = rows.setdefault(label, {"sieve": [], "extract": [], "full": []})
        if sieve is not None:
            r["sieve"].append(sieve)
        if extract is not None:
            r["extract"].append(extract)
        if full is not None:
            r["full"].append(full)

    if HAVE_NUMBA:  # compile outside the timed region
        extract_numba(sieve_numba(100))

    for run in range(1, runs + 1):
        print(f"  run {run}/{runs} ...", flush=True)

        flags, t_s = timed(sieve_bytearray, number)
        primes, t_e = timed(extract_bytearray, flags, number)
        counts["bytearray"] = check("bytearray", len(primes), number)
        add("bytearray (compress)", t_s, t_e, t_s + t_e)
        del primes

        if include_original:
            primes, t_e2 = timed(extract_bytearray_original, flags, number)
            check("bytearray original", len(primes), number)
            add("bytearray (original comprehension)", t_s, t_e2, t_s + t_e2)
            del primes
        del flags

        if HAVE_NUMBA:
            flags, t_s = timed(sieve_numba, number)
            primes, t_e = timed(extract_numba, flags)
            counts["numba"] = check("numba", len(primes), number)
            add("Numba njit", t_s, t_e, t_s + t_e)
            del flags, primes

        cnt, t_c = timed(sieve_parallel, number, "count")
        check("parallel count", cnt, number)
        primes, t_f = timed(sieve_parallel, number, "primes")
        counts["parallel"] = check("parallel", len(primes), number)
        add("multiprocessing (segmented)", sieve=t_c, full=t_f)
        del primes

    if len(set(counts.values())) != 1:
        raise SystemExit(f"Versions disagree: {counts}")
    return rows, next(iter(counts.values()))


def fmt(values):
    if not values:
        return "n/a"
    return f"{statistics.median(values):.3f} (min {min(values):.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("number", nargs="?", type=int, default=1_000_000_000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--include-original", action="store_true")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}, Numba: {'yes' if HAVE_NUMBA else 'not installed'}")
    print(f"N = {args.number:,}, runs = {args.runs}\n")

    rows, count = run_all(args.number, args.runs, args.include_original)

    print(f"\n{count:,} primes found (verified across versions)\n")
    print("Median seconds over runs (min in parentheses)\n")
    print("| Implementation | Sieve only | Extract primes | Sieve + extract |")
    print("|---|---|---|---|")
    for label, r in rows.items():
        print(f"| {label} | {fmt(r['sieve'])} | {fmt(r['extract'])} | {fmt(r['full'])} |")
    print("\nMultiprocessing 'sieve only' is a prime count; its extraction runs inside")
    print("the workers, so only the sieve-only and full times are reported.")


if __name__ == "__main__":
    main()
