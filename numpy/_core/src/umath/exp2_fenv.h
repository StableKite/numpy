#ifndef NUMPY_UMATH_EXP2_FENV_H
#define NUMPY_UMATH_EXP2_FENV_H
#include "numpy/npy_math.h"

#if defined(_WIN32)
/* UCRT exp2 may return zero for finite negative input without raising the
 * hardware underflow status. Keep its numeric result, and complete that
 * missing status for the scalar ufunc fallback (including half conversion
 * and legacy real loop callbacks). Infinity and NaN keep libm semantics.
 * These wrappers are private: public npymath headers/ABI are unchanged. */
static NPY_INLINE double
npy_ufunc_exp2_fenv(double value)
{
    double result = exp2(value);
    if (result == 0.0 && npy_isfinite(value) && value < 0.0) {
        npy_set_floatstatus_underflow();
    }
    return result;
}
static NPY_INLINE float
npy_ufunc_exp2f_fenv(float value)
{
    float result = exp2f(value);
    if (result == 0.0f && npy_isfinite(value) && value < 0.0f) {
        npy_set_floatstatus_underflow();
    }
    return result;
}
static NPY_INLINE npy_longdouble
npy_ufunc_exp2l_fenv(npy_longdouble value)
{
    npy_longdouble result = exp2l(value);
    if (result == 0.0L && npy_isfinite(value) && value < 0.0L) {
        npy_set_floatstatus_underflow();
    }
    return result;
}
#undef npy_exp2
#undef npy_exp2f
#undef npy_exp2l
#define npy_exp2 npy_ufunc_exp2_fenv
#define npy_exp2f npy_ufunc_exp2f_fenv
#define npy_exp2l npy_ufunc_exp2l_fenv
#endif
#endif
