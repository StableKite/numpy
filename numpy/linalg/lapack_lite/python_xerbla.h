#ifndef NUMPY_LINALG_PYTHON_XERBLA_H
#define NUMPY_LINALG_PYTHON_XERBLA_H
#include <Python.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed for the dynamic extent of the bound wrapper/ArrayMethod call. */
PyThreadState *npy_lapack_xerbla_set_owner(PyThreadState *owner);
#ifdef __cplusplus
}
#endif
#endif
