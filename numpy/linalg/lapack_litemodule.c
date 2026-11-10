/*This module contributed by Doug Heisterkamp
Modified by Jim Hugunin
More modifications by Jeff Whitaker
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/npy_common.h"
#include "numpy/interpreter_context.h"
#include "lapack_lite/python_xerbla.h"

typedef struct {
    NPY_SI_Context numpy;
    PyObject *error;
    int runtime_version;
} lapack_lite_state;

/* Borrowed only for one wrapper call; nested callbacks restore the old state. */
static _Thread_local lapack_lite_state *lapack_lite_active;
#define NPY_API_MODULE_CONTEXT (&lapack_lite_active->numpy)
#define NPY_API_MODULE_RUNTIME_VERSION (lapack_lite_active->runtime_version)
#include "numpy/arrayobject.h"
#include "npy_cblas.h"

#define FNAME(name) BLAS_FUNC(name)

typedef CBLAS_INT        fortran_int;

#ifdef HAVE_BLAS_ILP64

#if NPY_BITSOF_SHORT == 64
#define FINT_PYFMT       "h"
#elif NPY_BITSOF_INT == 64
#define FINT_PYFMT       "i"
#elif NPY_BITSOF_LONG == 64
#define FINT_PYFMT       "l"
#elif NPY_BITSOF_LONGLONG == 64
#define FINT_PYFMT       "L"
#else
#error No compatible 64-bit integer size. \
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform, or dont try to use ILP64 BLAS
#endif

#else
#define FINT_PYFMT       "i"
#endif

typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

extern fortran_int FNAME(dgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
                          double a[], fortran_int *lda, double b[], fortran_int *ldb,
                          double s[], double *rcond, fortran_int *rank,
                          double work[], fortran_int *lwork, fortran_int iwork[], fortran_int *info);

extern fortran_int FNAME(zgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
                          f2c_doublecomplex a[], fortran_int *lda,
                          f2c_doublecomplex b[], fortran_int *ldb,
                          double s[], double *rcond, fortran_int *rank,
                          f2c_doublecomplex work[], fortran_int *lwork,
                          double rwork[], fortran_int iwork[], fortran_int *info);

extern fortran_int FNAME(dgeqrf)(fortran_int *m, fortran_int *n, double a[], fortran_int *lda,
                          double tau[], double work[],
                          fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(zgeqrf)(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int *lda,
                          f2c_doublecomplex tau[], f2c_doublecomplex work[],
                          fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(dorgqr)(fortran_int *m, fortran_int *n, fortran_int *k, double a[], fortran_int *lda,
                          double tau[], double work[],
                          fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(zungqr)(fortran_int *m, fortran_int *n, fortran_int *k, f2c_doublecomplex a[],
                          fortran_int *lda, f2c_doublecomplex tau[],
                          f2c_doublecomplex work[], fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(xerbla)(char *srname, fortran_int *info);

#define LapackError (lapack_lite_active->error)

/* Bundled f2c has shared numeric scratch storage. Only the pure Fortran call
 * is locked: argument conversion may invoke Python and reenter this module. */
#if PY_VERSION_HEX >= 0x030d00b3
static PyMutex lapack_lite_call_lock = {0};
#define LOCK_LAPACK_LITE PyMutex_Lock(&lapack_lite_call_lock)
#define UNLOCK_LAPACK_LITE PyMutex_Unlock(&lapack_lite_call_lock)
#else
/* Older supported builds retain their GIL and the original call contract. */
#define LOCK_LAPACK_LITE
#define UNLOCK_LAPACK_LITE
#endif

#define TRY(E) if (!(E)) return NULL

static int
check_object(PyObject *ob, int t, char *obname,
                        char *tname, char *funname)
{
    if (!PyArray_Check(ob)) {
        PyErr_Format(LapackError,
                     "Expected an array for parameter %s in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    else if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject *)ob)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not contiguous in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    else if (!(PyArray_TYPE((PyArrayObject *)ob) == t)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not of type %s in lapack_lite.%s",
                     obname, tname, funname);
        return 0;
    }
    else if (PyArray_ISBYTESWAPPED((PyArrayObject *)ob)) {
        PyErr_Format(LapackError,
                     "Parameter %s has non-native byte order in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    else {
        return 1;
    }
}

#define CHDATA(p) ((char *) PyArray_DATA((PyArrayObject *)p))
#define SHDATA(p) ((short int *) PyArray_DATA((PyArrayObject *)p))
#define DDATA(p) ((double *) PyArray_DATA((PyArrayObject *)p))
#define FDATA(p) ((float *) PyArray_DATA((PyArrayObject *)p))
#define CDATA(p) ((f2c_complex *) PyArray_DATA((PyArrayObject *)p))
#define ZDATA(p) ((f2c_doublecomplex *) PyArray_DATA((PyArrayObject *)p))
#define IDATA(p) ((fortran_int *) PyArray_DATA((PyArrayObject *)p))

static PyObject *
lapack_lite_dgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int lapack_lite_status;
    fortran_int m;
    fortran_int n;
    fortran_int nrhs;
    PyObject *a;
    fortran_int lda;
    PyObject *b;
    fortran_int ldb;
    PyObject *s;
    double rcond;
    fortran_int rank;
    PyObject *work;
    PyObject *iwork;
    fortran_int lwork;
    fortran_int info;

    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT "O"
                          FINT_PYFMT "O" "d" FINT_PYFMT "O" FINT_PYFMT "O"
                          FINT_PYFMT ":dgelsd"),
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&iwork,&info));

    TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dgelsd"));
    TRY(check_object(b,NPY_DOUBLE,"b","NPY_DOUBLE","dgelsd"));
    TRY(check_object(s,NPY_DOUBLE,"s","NPY_DOUBLE","dgelsd"));
    TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dgelsd"));
#ifndef NPY_UMATH_USE_BLAS64_
    TRY(check_object(iwork,NPY_INT,"iwork","NPY_INT","dgelsd"));
#else
    TRY(check_object(iwork,NPY_INT64,"iwork","NPY_INT64","dgelsd"));
#endif

    LOCK_LAPACK_LITE;
    lapack_lite_status =
            FNAME(dgelsd)(&m,&n,&nrhs,DDATA(a),&lda,DDATA(b),&ldb,
                          DDATA(s),&rcond,&rank,DDATA(work),&lwork,
                          IDATA(iwork),&info);
    UNLOCK_LAPACK_LITE;
    if (PyErr_Occurred()) {
        return NULL;
    }

    return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:d,s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT "}"),
                         "dgelsd_",lapack_lite_status,"m",m,"n",n,"nrhs",nrhs,
                         "lda",lda,"ldb",ldb,"rcond",rcond,"rank",rank,
                         "lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_dgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
        fortran_int lapack_lite_status;
        fortran_int m, n, lwork;
        PyObject *a, *tau, *work;
        fortran_int lda;
        fortran_int info;

        TRY(PyArg_ParseTuple(args,
                             (FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT "OO"
                              FINT_PYFMT FINT_PYFMT ":dgeqrf"),
                             &m,&n,&a,&lda,&tau,&work,&lwork,&info));

        /* check objects and convert to right storage order */
        TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dgeqrf"));
        TRY(check_object(tau,NPY_DOUBLE,"tau","NPY_DOUBLE","dgeqrf"));
        TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dgeqrf"));

        LOCK_LAPACK_LITE;
        lapack_lite_status =
                FNAME(dgeqrf)(&m, &n, DDATA(a), &lda, DDATA(tau),
                              DDATA(work), &lwork, &info);
        UNLOCK_LAPACK_LITE;
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                              ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT "}"),
                             "dgeqrf_",
                             lapack_lite_status,"m",m,"n",n,"lda",lda,
                             "lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_dorgqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        fortran_int lapack_lite_status;
        fortran_int m, n, k, lwork;
        PyObject *a, *tau, *work;
        fortran_int lda;
        fortran_int info;

        TRY(PyArg_ParseTuple(args,
                             (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O"
                              FINT_PYFMT "OO" FINT_PYFMT FINT_PYFMT
                              ":dorgqr"),
                             &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dorgqr"));
        TRY(check_object(tau,NPY_DOUBLE,"tau","NPY_DOUBLE","dorgqr"));
        TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dorgqr"));
        LOCK_LAPACK_LITE;
        lapack_lite_status =
            FNAME(dorgqr)(&m, &n, &k, DDATA(a), &lda, DDATA(tau), DDATA(work),
                          &lwork, &info);
        UNLOCK_LAPACK_LITE;
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue("{s:i,s:i}","dorgqr_",lapack_lite_status,
                             "info",info);
}


static PyObject *
lapack_lite_zgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int lapack_lite_status;
    fortran_int m;
    fortran_int n;
    fortran_int nrhs;
    PyObject *a;
    fortran_int lda;
    PyObject *b;
    fortran_int ldb;
    PyObject *s;
    double rcond;
    fortran_int rank;
    PyObject *work;
    fortran_int lwork;
    PyObject *rwork;
    PyObject *iwork;
    fortran_int info;
    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT
                          "O" FINT_PYFMT "Od" FINT_PYFMT "O" FINT_PYFMT
                          "OO" FINT_PYFMT ":zgelsd"),
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&rwork,&iwork,&info));

    TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zgelsd"));
    TRY(check_object(b,NPY_CDOUBLE,"b","NPY_CDOUBLE","zgelsd"));
    TRY(check_object(s,NPY_DOUBLE,"s","NPY_DOUBLE","zgelsd"));
    TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zgelsd"));
    TRY(check_object(rwork,NPY_DOUBLE,"rwork","NPY_DOUBLE","zgelsd"));
#ifndef NPY_UMATH_USE_BLAS64_
    TRY(check_object(iwork,NPY_INT,"iwork","NPY_INT","zgelsd"));
#else
    TRY(check_object(iwork,NPY_INT64,"iwork","NPY_INT64","zgelsd"));
#endif

    LOCK_LAPACK_LITE;
    lapack_lite_status =
        FNAME(zgelsd)(&m,&n,&nrhs,ZDATA(a),&lda,ZDATA(b),&ldb,DDATA(s),&rcond,
                      &rank,ZDATA(work),&lwork,DDATA(rwork),IDATA(iwork),&info);
    UNLOCK_LAPACK_LITE;
    if (PyErr_Occurred()) {
        return NULL;
    }

    return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          "}"),
                         "zgelsd_",
                         lapack_lite_status,"m",m,"n",n,"nrhs",nrhs,"lda",lda,
                         "ldb",ldb,"rank",rank,"lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_zgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
        fortran_int lapack_lite_status;
        fortran_int m, n, lwork;
        PyObject *a, *tau, *work;
        fortran_int lda;
        fortran_int info;

        TRY(PyArg_ParseTuple(args,
                             (FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT "OO"
                              FINT_PYFMT "" FINT_PYFMT ":zgeqrf"),
                             &m,&n,&a,&lda,&tau,&work,&lwork,&info));

/* check objects and convert to right storage order */
        TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zgeqrf"));
        TRY(check_object(tau,NPY_CDOUBLE,"tau","NPY_CDOUBLE","zgeqrf"));
        TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zgeqrf"));

        LOCK_LAPACK_LITE;
        lapack_lite_status =
            FNAME(zgeqrf)(&m, &n, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                          &lwork, &info);
        UNLOCK_LAPACK_LITE;
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT
                              ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                              ",s:" FINT_PYFMT ",s:" FINT_PYFMT "}"),
                             "zgeqrf_",lapack_lite_status,"m",m,"n",n,"lda",lda,"lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_zungqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        fortran_int lapack_lite_status;
        fortran_int m, n, k, lwork;
        PyObject *a, *tau, *work;
        fortran_int lda;
        fortran_int info;

        TRY(PyArg_ParseTuple(args,
                             (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O"
                              FINT_PYFMT "OO" FINT_PYFMT "" FINT_PYFMT
                              ":zungqr"),
                             &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zungqr"));
        TRY(check_object(tau,NPY_CDOUBLE,"tau","NPY_CDOUBLE","zungqr"));
        TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zungqr"));


        LOCK_LAPACK_LITE;
        lapack_lite_status =
            FNAME(zungqr)(&m, &n, &k, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                          &lwork, &info);
        UNLOCK_LAPACK_LITE;
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT "}"),
                             "zungqr_",lapack_lite_status,
                             "info",info);
}


static PyObject *
lapack_lite_xerbla(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int info = -1;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;
    LOCK_LAPACK_LITE;
    FNAME(xerbla)("test", &info);
    UNLOCK_LAPACK_LITE;
    NPY_END_THREADS;

    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}


#define STR(x) #x
/* The bound module owns every Python object and API table used by the call. */
#define LAPACK_WRAPPER(name) \
static PyObject *lapack_lite_owned_##name(PyObject *self, PyObject *args) \
{ \
    lapack_lite_state *state = PyModule_GetState(self); \
    if (NPY_SI_Context_Check(&state->numpy) < 0) { \
        return NULL; \
    } \
    lapack_lite_state *previous = lapack_lite_active; \
    PyThreadState *previous_owner = npy_lapack_xerbla_set_owner(PyThreadState_Get()); \
    lapack_lite_active = state; \
    PyObject *result = lapack_lite_##name(self, args); \
    lapack_lite_active = previous; \
    npy_lapack_xerbla_set_owner(previous_owner); \
    return result; \
}
LAPACK_WRAPPER(dgelsd)
LAPACK_WRAPPER(dgeqrf)
LAPACK_WRAPPER(dorgqr)
LAPACK_WRAPPER(zgelsd)
LAPACK_WRAPPER(zgeqrf)
LAPACK_WRAPPER(zungqr)
LAPACK_WRAPPER(xerbla)
#define lameth(name) {STR(name), lapack_lite_owned_##name, METH_VARARGS, NULL}
static struct PyMethodDef lapack_lite_module_methods[] = {
    lameth(dgelsd),
    lameth(dgeqrf),
    lameth(dorgqr),
    lameth(zgelsd),
    lameth(zgeqrf),
    lameth(zungqr),
    lameth(xerbla),
    { NULL,NULL,0, NULL}
};

static int
lapack_lite_traverse(PyObject *m, visitproc visit, void *arg)
{
    lapack_lite_state *state = PyModule_GetState(m);
    Py_VISIT(state->error);
    return NPY_SI_Context_Traverse(&state->numpy, visit, arg);
}

static int
lapack_lite_clear(PyObject *m)
{
    lapack_lite_state *state = PyModule_GetState(m);
    Py_CLEAR(state->error);
    NPY_SI_Context_Clear(&state->numpy);
    return 0;
}

static void
lapack_lite_free(void *m)
{
    lapack_lite_clear((PyObject *)m);
}

static int
lapack_lite_exec(PyObject *m)
{
    lapack_lite_state *state = PyModule_GetState(m);
    if (state->numpy.owner != NULL) {
        return 0;
    }
    if (NPY_SI_Context_Import(&state->numpy) < 0) {
        return -1;
    }
    lapack_lite_state *previous = lapack_lite_active;
    lapack_lite_active = state;
    state->runtime_version = (int)PyArray_GetNDArrayCFeatureVersion();
    lapack_lite_active = previous;
    state->error = PyErr_NewException("numpy.linalg.lapack_lite.LapackError", NULL, NULL);
    if (state->error == NULL || PyModule_AddObjectRef(m, "LapackError", state->error) < 0) {
        return -1;
    }
#ifdef HAVE_BLAS_ILP64
    return PyModule_AddObjectRef(m, "_ilp64", Py_True);
#else
    return PyModule_AddObjectRef(m, "_ilp64", Py_False);
#endif
}

static struct PyModuleDef_Slot lapack_lite_slots[] = {
    {Py_mod_exec, lapack_lite_exec},
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
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "lapack_lite",
    .m_size = sizeof(lapack_lite_state),
    .m_methods = lapack_lite_module_methods,
    .m_slots = lapack_lite_slots,
    .m_traverse = lapack_lite_traverse,
    .m_clear = lapack_lite_clear,
    .m_free = lapack_lite_free,
};

PyMODINIT_FUNC PyInit_lapack_lite(void) {
    return PyModuleDef_Init(&moduledef);
}
