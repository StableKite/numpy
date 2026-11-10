#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "_owned_test_context.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include <math.h>


/*
 * struct_ufunc_test.c
 * This is the C code for creating your own
 * NumPy ufunc for a structured array dtype.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static void add_uint64_triplet(char **args,
                               npy_intp const *dimensions,
                               npy_intp const* steps,
                               void* data)
{
    npy_intp i;
    npy_intp is1=steps[0];
    npy_intp is2=steps[1];
    npy_intp os=steps[2];
    npy_intp n=dimensions[0];
    npy_uint64 *x, *y, *z;

    char *i1=args[0];
    char *i2=args[1];
    char *op=args[2];

    for (i = 0; i < n; i++) {

        x = (npy_uint64*)i1;
        y = (npy_uint64*)i2;
        z = (npy_uint64*)op;

        z[0] = x[0] + y[0];
        z[1] = x[1] + y[1];
        z[2] = x[2] + y[2];

        i1 += is1;
        i2 += is2;
        op += os;
    }
}

static PyObject*
register_fail(PyObject* NPY_UNUSED(self), PyObject* NPY_UNUSED(args))
{
    PyObject *add_triplet;
    PyObject *dtype_dict;
    PyArray_Descr *dtype;
    PyArray_Descr *dtypes[3];
    int retval;

    add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                    PyUFunc_None, "add_triplet",
                                    "add_triplet_docstring", 0);

    if (add_triplet == NULL) {
        return NULL;
    }
    dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
                               "f0", "u8", "f1", "u8", "f2", "u8");
    if (dtype_dict == NULL || !PyArray_DescrConverter(dtype_dict, &dtype)) {
        Py_XDECREF(dtype_dict);
        Py_DECREF(add_triplet);
        return NULL;
    }
    Py_DECREF(dtype_dict);

    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    retval = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);

    if (retval < 0) {
        Py_DECREF(add_triplet);
        Py_DECREF(dtype);
        return NULL;
    }
    retval = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);
    Py_DECREF(add_triplet);
    Py_DECREF(dtype);
    if (retval < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
owned_register_fail(PyObject *self, PyObject *args)
{
    owned_test_state *state = PyModule_GetState(self);
    if (NPY_SI_Context_Check(&state->numpy) < 0) {
        return NULL;
    }
    owned_test_state *previous = owned_test_active;
    owned_test_active = state;
    PyObject *result = register_fail(self, args);
    owned_test_active = previous;
    return result;
}

static PyMethodDef StructUfuncTestMethods[] = {
    {"register_fail",
        owned_register_fail,
        METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static int
initialize_ufuncs(PyObject *m)
{
    PyObject *ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
            PyUFunc_None, "add_triplet", NULL, 0);
    if (ufunc == NULL) {
        return -1;
    }
    PyObject *dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
            "f0", "u8", "f1", "u8", "f2", "u8");
    PyArray_Descr *dtype = NULL;
    if (dtype_dict == NULL || !PyArray_DescrConverter(dtype_dict, &dtype)) {
        Py_XDECREF(dtype_dict);
        Py_DECREF(ufunc);
        return -1;
    }
    Py_DECREF(dtype_dict);
    PyArray_Descr *dtypes[3] = {dtype, dtype, dtype};
    int result = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)ufunc,
            dtype, add_uint64_triplet, dtypes, NULL);
    Py_DECREF(dtype);
    if (result == 0) {
        result = PyModule_AddObjectRef(m, "add_triplet", ufunc);
    }
    Py_DECREF(ufunc);
    return result;
}

static int
owned_test_exec(PyObject *module)
{
    if (owned_test_initialize(module) < 0) {
        return -1;
    }
    owned_test_state *previous = owned_test_active;
    owned_test_active = PyModule_GetState(module);
    int result = initialize_ufuncs(module);
    owned_test_active = previous;
    return result;
}
static PyModuleDef_Slot owned_test_slots[] = {
    {Py_mod_exec, owned_test_exec},
#if PY_VERSION_HEX >= 0x030f0000
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#elif PY_VERSION_HEX >= 0x030c0000
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};
static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_struct_ufunc_tests",
    .m_size = sizeof(owned_test_state),
    .m_methods = StructUfuncTestMethods,
    .m_slots = owned_test_slots,
    .m_traverse = owned_test_traverse,
    .m_clear = owned_test_clear,
    .m_free = owned_test_free
};
PyMODINIT_FUNC PyInit__struct_ufunc_tests(void)
{
    return PyModuleDef_Init(&moduledef);
}
