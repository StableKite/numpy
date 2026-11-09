/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*
 *  Main implementation file.
 *
 *  Copyright (C) 2004-2018 Max-Planck-Society
 *  \author Martin Reinecke
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
#include "numpy/interpreter_context.h"

#include "npy_config.h"

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft/pocketfft_hdronly.h"

/*
 * In order to ensure that C++ exceptions are converted to Python
 * ones before crossing over to the C machinery, we must catch them.
 * This template can be used to wrap a C++ written ufunc to do this via:
 *      wrap_legacy_cpp_ufunc<cpp_ufunc>
 */
template<PyUFuncGenericFunction cpp_ufunc>
static void
wrap_legacy_cpp_ufunc(char **args, npy_intp const *dimensions,
                      npy_intp const *steps, void *func)
{
    NPY_ALLOW_C_API_DEF
    try {
        cpp_ufunc(args, dimensions, steps, func);
    }
    catch (std::bad_alloc& e) {
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    catch (const std::exception& e) {
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        NPY_DISABLE_C_API;
    }
}

/*
 * Transfer to and from a contiguous buffer.
 * copy_input: copy min(nin, n) elements from input to buffer and zero rest.
 * copy_output: copy n elements from buffer to output.
 */
template <typename T>
static inline void
copy_input(char *in, npy_intp step_in, size_t nin,
           T buff[], size_t n)
{
    size_t ncopy = nin <= n ? nin : n;
    char *ip = in;
    size_t i;
    for (i = 0; i < ncopy; i++, ip += step_in) {
      buff[i] = *(T *)ip;
    }
    for (; i < n; i++) {
      buff[i] = 0;
    }
}

template <typename T>
static inline void
copy_output(T buff[], char *out, npy_intp step_out, size_t n)
{
    char *op = out;
    for (size_t i = 0; i < n; i++, op += step_out) {
        *(T *)op = buff[i];
    }
}

/*
 * Gufunc loops calling the pocketfft code.
 */
template <typename T>
static void
fft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    bool direction = *((bool *)func); /* pocketfft::FORWARD or BACKWARD */

    assert (nout > 0);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * For the common case of nin >= nout, fixed factor, and suitably sized
     * outer loop, we call pocketfft directly to benefit from its vectorization.
     * (For nin>nout, this just removes the extra input points, as required;
     * the vlen constraint avoids compiling extra code for longdouble, which
     * cannot be vectorized so does not benefit.)
     */
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= nout && sf == 0) {
        std::vector<size_t> shape = { n_outer, nout };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        std::vector<size_t> axes = { 1 };
        pocketfft::c2c(shape, strides_in, strides_out, axes, direction,
                       (std::complex<T> *)ip, (std::complex<T> *)op, *(T *)fp);
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<T>>(nout);
    auto buffered = (step_out != sizeof(std::complex<T>));
    pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
        if (ip != (char*)op_or_buff) {
            copy_input(ip, step_in, nin, op_or_buff, nout);
        }
        plan->exec((pocketfft::detail::cmplx<T> *)op_or_buff, *(T *)fp, direction);
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    return;
}

template <typename T>
static void
rfft_impl(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *func, size_t npts)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];

    assert (nout > 0 && nout == npts / 2 + 1);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * Call pocketfft directly if vectorization is possible.
     */
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= npts && sf == 0) {
        std::vector<size_t> shape_in = { n_outer, npts };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        std::vector<size_t> axes = { 1 };
        pocketfft::r2c(shape_in, strides_in, strides_out, axes, pocketfft::FORWARD,
                       (T *)ip, (std::complex<T> *)op, *(T *)fp);
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(npts);
    auto buffered = (step_out != sizeof(std::complex<T>));
    pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
    auto nin_used = nin <= npts ? nin : npts;
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
        /*
         * The internal pocketfft routines work in-place and for real
         * transforms the frequency data thus needs to be compressed, using
         * that there will be no imaginary component for the zero-frequency
         * item (which is the sum of all inputs and thus has to be real), nor
         * one for the Nyquist frequency for even number of points.
         * Pocketfft uses FFTpack order, R0,R1,I1,...Rn-1,In-1,Rn[,In] (last
         * for npts odd only). To make unpacking easy, we place the real data
         * offset by one in the buffer, so that we just have to move R0 and
         * create I0=0. Note that copy_input will zero the In component for
         * even number of points.
         */
        copy_input(ip, step_in, nin_used, &((T *)op_or_buff)[1], nout*2 - 1);
        plan->exec(&((T *)op_or_buff)[1], *(T *)fp, pocketfft::FORWARD);
        op_or_buff[0] = op_or_buff[0].imag();  // I0->R0, I0=0
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    return;
}

/*
 * For the forward real, we cannot know what the requested number of points is
 * just based on the number of points in the complex output array (e.g., 10
 * and 11 real input points both lead to 6 complex output points), so we
 * define versions for both even and odd number of points.
 */
template <typename T>
static void
rfft_n_even_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    size_t nout = (size_t)dimensions[2];
    assert (nout > 0);
    size_t npts = 2 * nout - 2;
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void
rfft_n_odd_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    size_t nout = (size_t)dimensions[2];
    assert (nout > 0);
    size_t npts = 2 * nout - 1;
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    ptrdiff_t si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    ptrdiff_t step_in = steps[3], step_out = steps[4];

    assert(nout > 0);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * Call pocketfft directly if vectorization is possible.
     */
    size_t npts_in = nout / 2 + 1;
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= npts_in && sf == 0) {
        std::vector<size_t> axes = { 1 };
        std::vector<size_t> shape_out = { n_outer, nout };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        pocketfft::c2r(shape_out, strides_in, strides_out, axes, pocketfft::BACKWARD,
                       (std::complex<T> *)ip, (T *)op, *(T *)fp);
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(nout);
    auto buffered = (step_out != sizeof(T));
    pocketfft::detail::arr<T> buff(buffered ? nout : 0);
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        T *op_or_buff = buffered ? buff.data() : (T *)op;
        /*
         * Pocket_fft works in-place and for inverse real transforms the
         * frequency data thus needs to be compressed, removing the imaginary
         * component of the zero-frequency item (which is the sum of all
         * inputs and thus has to be real), as well as the imaginary component
         * of the Nyquist frequency for even number of points. We thus copy
         * the data to the buffer in the following order (also used by
         * FFTpack): R0,R1,I1,...Rn-1,In-1,Rn[,In] (last for npts odd only).
         */
        op_or_buff[0] = ((T *)ip)[0];  /* copy R0 */
        if (nout > 1) {
            /*
             * Copy R1,I1... up to Rn-1,In-1 if possible, stopping earlier
             * if not all the input points are needed or if the input is short
             * (in the latter case, zeroing after).
             */
            copy_input(ip + step_in, step_in, nin - 1,
                       (std::complex<T> *)&op_or_buff[1], (nout - 1) / 2);
            /* For even nout, we still need to set Rn. */
            if (nout % 2 == 0) {
                op_or_buff[nout - 1] = (nout / 2 >= nin) ? (T)0 :
                    ((T *)(ip + (nout / 2) * step_in))[0];
            }
        }
        plan->exec(op_or_buff, *(T *)fp, pocketfft::BACKWARD);
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    return;
}

static PyUFuncGenericFunction fft_functions[] = {
    wrap_legacy_cpp_ufunc<fft_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<fft_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<fft_loop<npy_longdouble>>
};
static const char fft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};
static void *const fft_data[] = {
    (void*)&pocketfft::FORWARD,
    (void*)&pocketfft::FORWARD,
    (void*)&pocketfft::FORWARD
};
static void *const ifft_data[] = {
    (void*)&pocketfft::BACKWARD,
    (void*)&pocketfft::BACKWARD,
    (void*)&pocketfft::BACKWARD
};

static PyUFuncGenericFunction rfft_n_even_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_longdouble>>
};
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_longdouble>>
};
static const char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

static PyUFuncGenericFunction irfft_functions[] = {
    wrap_legacy_cpp_ufunc<irfft_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<irfft_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<irfft_loop<npy_longdouble>>
};
static const char irfft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE
};

typedef struct {
    NPY_SI_Context numpy;
} pocketfft_module_state;

/* The public ArrayMethod keeps the caller attached on entry. Detach only for
 * the pure C++ computation and restore that exact thread state on every exit.
 * PyGILState_Ensure cannot select a detached subinterpreter's owning state. */
template<PyUFuncGenericFunction cpp_ufunc, bool forward>
static int
owned_fft_loop(PyArrayMethod_Context *, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides, NpyAuxData *)
{
    bool direction = forward;
    PyThreadState *caller = PyEval_SaveThread();
    try {
        cpp_ufunc(const_cast<char **>(args), dimensions, strides, &direction);
    }
    catch (const std::bad_alloc&) {
        PyEval_RestoreThread(caller);
        PyErr_SetString(PyExc_MemoryError, "pocketfft C++ allocation failed");
        return -1;
    }
    catch (const std::exception& error) {
        PyEval_RestoreThread(caller);
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return -1;
    }
    catch (...) {
        PyEval_RestoreThread(caller);
        PyErr_SetString(PyExc_RuntimeError, "unknown pocketfft C++ exception");
        return -1;
    }
    PyEval_RestoreThread(caller);
    return 0;
}

static int
add_gufuncs(PyObject *dictionary, NPY_SI_Context *context)
{
    typedef PyObject *(*constructor_func)(PyUFuncGenericFunction *, void *const *,
            const char *, int, int, int, int, const char *, const char *, int, const char *);
    typedef int (*add_loop_func)(PyObject *, PyArrayMethod_Spec *);
    auto constructor = (constructor_func)context->ufunc_api[31];
    auto add_loop = (add_loop_func)context->ufunc_api[43];
    struct definition {
        const char *name, *doc, *signature;
        PyUFuncGenericFunction *legacy_functions;
        void *const *legacy_data;
        const char *types;
        PyArrayMethod_StridedLoop *loops[3];
        bool real_input, real_output;
    } definitions[] = {
        {"fft", "complex forward FFT\n", "(n),()->(m)", fft_functions, fft_data, fft_types,
            {owned_fft_loop<fft_loop<npy_double>, true>, owned_fft_loop<fft_loop<npy_float>, true>,
             owned_fft_loop<fft_loop<npy_longdouble>, true>}, false, false},
        {"ifft", "complex backward FFT\n", "(m),()->(n)", fft_functions, ifft_data, fft_types,
            {owned_fft_loop<fft_loop<npy_double>, false>, owned_fft_loop<fft_loop<npy_float>, false>,
             owned_fft_loop<fft_loop<npy_longdouble>, false>}, false, false},
        {"rfft_n_even", "real forward FFT for even n\n", "(n),()->(m)", rfft_n_even_functions, NULL, rfft_types,
            {owned_fft_loop<rfft_n_even_loop<npy_double>, true>, owned_fft_loop<rfft_n_even_loop<npy_float>, true>,
             owned_fft_loop<rfft_n_even_loop<npy_longdouble>, true>}, true, false},
        {"rfft_n_odd", "real forward FFT for odd n\n", "(n),()->(m)", rfft_n_odd_functions, NULL, rfft_types,
            {owned_fft_loop<rfft_n_odd_loop<npy_double>, true>, owned_fft_loop<rfft_n_odd_loop<npy_float>, true>,
             owned_fft_loop<rfft_n_odd_loop<npy_longdouble>, true>}, true, false},
        {"irfft", "real backward FFT\n", "(m),()->(n)", irfft_functions, NULL, irfft_types,
            {owned_fft_loop<irfft_loop<npy_double>, false>, owned_fft_loop<irfft_loop<npy_float>, false>,
             owned_fft_loop<irfft_loop<npy_longdouble>, false>}, false, true},
    };
    /* Existing public dtype API slots: double/float/longdouble and their
     * complex partners. Addresses are borrowed from this module's owner. */
    const int real_slots[3] = {344, 343, 345};
    const int complex_slots[3] = {347, 346, 348};
    for (auto &definition : definitions) {
        PyObject *function = constructor(NULL, NULL, NULL, 0, 2, 1, PyUFunc_None,
                definition.name, definition.doc, 0, definition.signature);
        if (function == NULL) {
            return -1;
        }
        for (int index = 0; index < 3; index++) {
            auto real = (PyArray_DTypeMeta *)context->array_api[real_slots[index]];
            auto complex = (PyArray_DTypeMeta *)context->array_api[complex_slots[index]];
            PyArray_DTypeMeta *dtypes[3] = {
                definition.real_input ? real : complex, real,
                definition.real_output ? real : complex};
            PyType_Slot slots[] = {
                {NPY_METH_strided_loop, (void *)definition.loops[index]}, {0, NULL}};
            PyArrayMethod_Spec spec = {definition.name, 2, 1, NPY_NO_CASTING,
                NPY_METH_REQUIRES_PYAPI, dtypes, slots};
            if (add_loop(function, &spec) < 0) {
                Py_DECREF(function);
                return -1;
            }
        }
        /* Preserve the old native ufunc's legacy type/signature introspection.
         * Normal dispatch uses the owned modern loops registered above. */
        auto ufunc = (PyUFuncObject *)function;
        ufunc->functions = definition.legacy_functions;
        ufunc->data = definition.legacy_data;
        ufunc->types = definition.types;
        ufunc->ntypes = 3;
        int result = PyDict_SetItemString(dictionary, definition.name, function);
        Py_DECREF(function);
        if (result < 0) {
            return -1;
        }
    }
    return 0;
}

static int
pocketfft_traverse(PyObject *module, visitproc visit, void *arg)
{
    auto state = (pocketfft_module_state *)PyModule_GetState(module);
    return NPY_SI_Context_Traverse(&state->numpy, visit, arg);
}

static int
pocketfft_clear(PyObject *module)
{
    auto state = (pocketfft_module_state *)PyModule_GetState(module);
    NPY_SI_Context_Clear(&state->numpy);
    return 0;
}

static void
pocketfft_free(void *module)
{
    pocketfft_clear((PyObject *)module);
}

#if PY_VERSION_HEX < 0x030f0000
static int module_loaded = 0;
#endif

static int
_pocketfft_umath_exec(PyObject *m)
{
#if PY_VERSION_HEX < 0x030f0000
    // Preserve older runtimes' single-module contract.
    if (module_loaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }
    module_loaded = 1;
#endif

    auto state = (pocketfft_module_state *)PyModule_GetState(m);
    if (NPY_SI_Context_Import(&state->numpy) < 0) {
        return -1;
    }
    PyObject *dictionary = PyModule_GetDict(m);
    if (dictionary == NULL || add_gufuncs(dictionary, &state->numpy) < 0) {
        return -1;
    }

    return 0;
}

static struct PyModuleDef_Slot _pocketfft_umath_slots[] = {
    {Py_mod_exec, (void*)_pocketfft_umath_exec},
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
#if PY_VERSION_HEX >= 0x030f0000
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#else
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    "_pocketfft_umath",     /* m_name */
    NULL,                   /* m_doc */
    sizeof(pocketfft_module_state), /* m_size */
    NULL,                   /* m_methods */
    _pocketfft_umath_slots, /* m_slots */
    pocketfft_traverse,
    pocketfft_clear,
    pocketfft_free,
};

PyMODINIT_FUNC PyInit__pocketfft_umath(void) {
    return PyModuleDef_Init(&moduledef);
}
