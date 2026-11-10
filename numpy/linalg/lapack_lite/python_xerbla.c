#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>

#include "numpy/npy_common.h"
#include "npy_cblas.h"
#include "python_xerbla.h"

static _Thread_local PyThreadState *npy_lapack_xerbla_owner;

PyThreadState *
npy_lapack_xerbla_set_owner(PyThreadState *owner)
{
    PyThreadState *previous = npy_lapack_xerbla_owner;
    npy_lapack_xerbla_owner = owner;
    return previous;
}

/*
  From the original manpage:
  --------------------------
  XERBLA is an error handler for the LAPACK routines.
  It is called by an LAPACK routine if an input parameter has an invalid value.
  A message is printed and execution stops.

  Instead of printing a message and stopping the execution, a
  ValueError is raised with the message.

  Parameters:
  -----------
  srname: Subroutine name to use in error message, maximum six characters.
          Spaces at the end are skipped.
  info: Number of the invalid parameter.
*/

CBLAS_INT BLAS_FUNC(xerbla)(char *srname, CBLAS_INT *info)
{
        static const char format[] = "On entry to %.*s" \
                " parameter number %d had an illegal value";
        char buf[sizeof(format) + 6 + 4];   /* 6 for name, 4 for param. num. */

        int len = 0; /* length of subroutine name*/
#if PY_VERSION_HEX >= 0x030f0000
        PyThreadState *attached = PyThreadState_GetUnchecked();
        int restore = attached == NULL;
        if (restore) {
            if (npy_lapack_xerbla_owner == NULL) {
                Py_FatalError("LAPACK error callback has no captured Python caller");
            }
            PyEval_RestoreThread(npy_lapack_xerbla_owner);
        }
#else
        PyGILState_STATE save;
#endif

        while( len<6 && srname[len]!='\0' )
                len++;
        while( len && srname[len-1]==' ' )
                len--;
#if PY_VERSION_HEX < 0x030f0000
        save = PyGILState_Ensure();
#endif
        PyOS_snprintf(buf, sizeof(buf), format, len, srname, (int)*info);
        PyErr_SetString(PyExc_ValueError, buf);
#if PY_VERSION_HEX >= 0x030f0000
        if (restore) {
            PyThreadState *detached = PyEval_SaveThread();
            assert(detached == npy_lapack_xerbla_owner);
        }
#else
        PyGILState_Release(save);
#endif

        return 0;
}
