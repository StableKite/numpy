StableKite CPython 3.15t engineering component
============================================

This branch descends from the actual NumPy 2.5.4 release. Its local version
2.5.4+stablekite.1 identifies the modified native implementation and allows a
normal pip installation to replace stock 2.5.4, including in a virtualenv
which inherits system site packages. It remains compatible with the
``numpy==2.5.4`` and ``numpy<2.6`` requirement specifiers.

Qualified interpreter ownership covers core arrays and dtypes, linalg ufuncs,
FFT, and all nine random extensions on stock CPython 3.15t x64 Windows and
Linux/glibc. Each native module owns its interpreter context and heap types;
detached numeric calls restore their actual caller for error handling.
Real child imports, independent classes and streams, native error recovery,
natural GC, callback failures/retry/reentry, and physical child closure were
tested with the GIL naturally disabled. Existing 2.5.4 main C binaries are
retained. Legacy clients not prepared for an owned child API fail honestly.

This is an engineering component, not a claim that every private native
namespace supports children. In particular, lapack_lite and native testing
extensions retain their existing guards. Wider shutdown/failure coverage,
older Python source-build controls, and a pre-existing Windows exp2 extreme
underflow status issue remain open. No new skip/xfail hides those boundaries.

Ordinary Git-pip installation on the qualified ABI downloads the wheel named
in stablekite-wheels.json. The backend verifies its size, full SHA-256,
package and runtime versions, actual native source commit, platform tag and
native binary hashes. The source commit used to build each wheel is explicit;
subsequent manifest-only delivery commits do not pretend to rebuild C code.
The Linux wheel's required glibc version is checked before use.

To compile this checked-in implementation instead, pass
``-Cstablekite-source-build=true`` to pip or python -m build. This delegates
to NumPy's original meson-python build. The prepared random adapter requires
Cython 3.3.1 and validates the generated-code landmarks; it does not silently
accept another generator. Other ABIs use the source backend and are not
covered by the CPython 3.15t binary qualification.

The upstream main branch is preserved. Publication and clean ordinary pip
upgrade checks are recorded separately from local native component results.
