# Sieve of Eratosthenes in Python: bytearray vs Numba vs multiprocessing

Three implementations of an odd-only Sieve of Eratosthenes, benchmarked up to 1,000,000,000 (50,847,534 primes). Every version is verified against the known prime count, and every timing is split into phases so the comparison is fair.

## Results (N = 1,000,000,000)

Median of 5 runs, seconds (best run in parentheses).

| Implementation | Sieve only | Extract primes | Sieve + extract |
|---|---|---|---|
| Pure Python `bytearray` (slice assignment, `compress` extraction) | 3.395 (3.266) | 5.291 (5.210) | 8.606 (8.593) |
| Pure Python `bytearray`, original list comprehension | 3.395 (3.266) | 11.939 (11.738) | 15.274 (15.132) |
| Numba `njit` (single thread) | 3.230 (2.936) | 0.502 (0.497) | 3.726 (3.433) |
| Multiprocessing, segmented `bytearray` | 1.211 (1.184) | n/a | 1.540 (1.537) |

- **Sieve only** builds the flag array (or counts primes, for the multiprocessing version).
- **Extract primes** converts flags into the actual list of primes. The Python version returns a list of ints, the Numba and multiprocessing versions return a NumPy `int64` array.
- **Multiprocessing** timings include process-pool startup and transferring results back to the parent. Extraction happens inside the workers, so only the sieve-only (count) and full times are reported.

**System:** Intel Core 7 240H (6P + 4E cores, 16 threads), 32 GB DDR5-5600, Windows 11, Python 3.14.7, NumPy 2.5.4, Numba 0.68.0.

Reproduce with:

```
python sieve_bench.py                    # N = 1e9, 5 runs
python sieve_bench.py --include-original # also time the original list comprehension
```

## What the benchmark shows

- **The sieve phase is essentially tied between `bytearray` and Numba.** 3.40 s vs 3.23 s is a gap of about 5%, similar in size to the run-to-run variation. Both perform the same strided byte writes over a ~500 MB array.
- **Extraction is where single-process versions differ.** Numba's count-then-fill loop takes 0.50 s. Building the list of 50.8 million Python ints takes 5.29 s with `itertools.compress`, about 10x slower, and 11.94 s with the original list comprehension, about 2.3x slower than `compress`.
- **End to end, Numba is about 2.3x faster than the Python version** (3.73 s vs 8.61 s), and about 4.1x faster than the original code (15.27 s).
- **The segmented multiprocessing version is the fastest overall at 1.54 s.** That is about 2.4x faster than Numba and about 5.6x faster than the `compress` version end to end. Its sieve phase (counting only) is about 2.7x faster than the single-process versions, using 16 hardware threads, so scaling is real but well short of linear. This benchmark does not isolate why.
- **The original "3.46 s for all primes" figure** matches the sieve-only column (3.40 s here). It did not include building the list of primes.

## How it works

### Odd-only sieve
Index `i` in the flag array represents the odd number `2*i + 1`, which halves memory (500 MB instead of 1 GB for N = 1e9). For each prime `p` up to sqrt(N), multiples starting at `p*p` are cleared with a stride of `p` in index space.

### Why slice assignment is fast in pure Python
`flags[start:size:p] = bytes(n)` runs as a single C loop inside CPython, so the interpreter is not involved per element. This is why the `bytearray` version is far faster than a Python `for` loop. It is a scalar strided write loop, one byte per step, not a SIMD or vectorized operation. Only contiguous (step 1) slices can use `memcpy`-style bulk copies.

### Why Numba is in the same range for the sieve
Numba compiles the inner `for idx in range(start, size, p)` loop to the same kind of strided byte stores. Both versions touch the whole 500 MB array once per small prime, so for large N both are limited by cache misses and memory access patterns rather than by instruction count. Differences between them come mostly from the extraction phase and from implementation details, not from "compiled vs built-in" as such.

### Extraction
- **Python:** `itertools.compress(range(1, 2*len(flags), 2), flags)` keeps the scan in C. The original `[2*i+1 for i, f in enumerate(flags) if f]` runs a Python-level loop over ~500 million entries and is much slower. Either way, the result is a list holding ~50.8 million separate int objects, which needs well over 1 GB of memory (the list's pointers alone are ~400 MB, and each int above 256 is its own ~28-byte object).
- **Numba:** a counting pass followed by a fill pass writes directly into a single `int64` array (~400 MB).

### Multiprocessing version
This uses `ProcessPoolExecutor` (separate processes, not threads, so the GIL is not the limiting factor). The main process sieves the base primes up to sqrt(N), then each worker sieves one segment of 256 Ki odd numbers and returns its primes as a NumPy array. Segments are small enough to fit in L2 cache, which is the main reason a segmented sieve can beat a single large-array sieve.

## Corrections to earlier versions of this README

Earlier versions of this project made claims that were not supported by the measurements. They have been removed:

- **Timing scope.** The original pure Python timer stopped before the list of primes was built, while the Numba timer included building the result. The two numbers were not comparable. All timings here report sieve and extraction separately.
- **Extraction time.** "All primes in 3.46 s" described only the sieve phase. Building the 50.8M-element list is a separate cost and was not included.
- **"SIMD saturation" / "block blitting".** Strided slice assignment is a scalar C loop, not a vector operation.
- **"Python beats Numba."** The sieve loops are essentially equivalent. The earlier gap was explained by the Numba timer including extraction (plus extra temporary arrays in the old Numba code) and the Python timer not.
- **"Multithreaded version loses due to L3 contention."** That version used processes, not threads, and the timer included result extraction and transfer. No cache-contention measurement was made.
- **"RAM bandwidth is the ultimate bottleneck."** Not demonstrated. A flat sieve over a 500 MB array is largely limited by cache misses, and cache-blocked (segmented) sieves are generally much faster than this approach. Dedicated implementations such as `primesieve` sieve 1e9 far faster than any version here.
- **"Prefetchers stream the list in under half a second."** Not supported; creating the list takes far longer than that.
- **Sub-millisecond micro-benchmarks** (the 1,000,000 "percent speedup" figures) were dominated by timer noise and are no longer reported.
