StableKite CPython 3.15t NumPy component
======================================

This branch descends from the actual NumPy 2.5.4 release. The real source
version 2.5.4+stablekite.2 identifies its modified implementation and permits
ordinary pip upgrades from stock 2.5.4. It satisfies numpy==2.5.4 and numpy<2.6
requirement specifiers without renaming a newer upstream NumPy release.

Interpreter ownership covers all 19 native modules, including lapack_lite,
SIMD, the native test extensions, FFT and all nine random extensions.
Each module owns its interpreter context and heap types. Detached numerical
calls restore the actual calling thread state when setting Python errors.
Parser caches contain C signature metadata; Python keyword objects belong
to a particular call. Legacy test loops retain call-scoped native owner data.
Rational scalar/DType registration and canonical module reuse belong to the
interpreter. DType teardown respects the builtin type deallocator's GC handoff.

Qualification uses stock CPython 3.15.2t x64 on Windows and Linux/glibc with
the GIL naturally disabled. Both import orders, real child closure, concurrent
workers, nested Python callbacks, failure/retry paths, late native values and
natural module/array GC are exercised. Original core, dtype, string, ufunc,
SIMD, linalg, FFT and random suites run without new skips or xfails. Windows
exp2 extreme-underflow status and linspace converter ownership are corrected.

Existing main-interpreter C clients compiled against genuine NumPy 2.5.4
headers are retained as binaries. Child C clients must opt in to the owned
interpreter API; unprepared legacy clients are rejected. Source-build controls
cover Python 3.12, 3.13, 3.14 and 3.14t in their natural GIL mode. The older
runtime subinterpreter guards remain closed: child support targets 3.15t.

Ordinary Git-pip installation on the qualified ABI downloads the wheel in
stablekite-wheels.json after checking its SHA-256, size, platform, package and
runtime versions, actual native source commit and all native binary hashes.
Linux minimum glibc is checked. A later manifest-only carrier commit records
the completed native build; it does not claim to rebuild native source.

Pass -Cstablekite-source-build=true to compile the checked-in implementation
through the original meson-python backend. The random adapter validates its
generated Cython 3.3.1 landmarks. Other ABIs use source builds and have no claim
to the qualified CPython 3.15t binary or platform coverage.

The upstream main branch and previous immutable release artifacts remain
unchanged. NumPy qualification does not imply Greenlet qualification or fix
independent defects in the CPython concurrent.interpreters Python wrapper.
