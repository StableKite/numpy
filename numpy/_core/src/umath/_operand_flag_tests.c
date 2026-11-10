#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "_owned_test_context.h"
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include "numpy/npy_3kcompat.h"
#include <math.h>
#include <structmember.h>


static PyMethodDef TestMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void
inplace_add(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp in2_step = steps[1];

    for (i = 0; i < n; i++) {
        (*(npy_intp *)in1) = *(npy_intp*)in1 + *(npy_intp*)in2;
        in1 += in1_step;
        in2 += in2_step;
    }
}


/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&inplace_add};

/* These are the input and return dtypes of logit.*/
static const char types[2] = {NPY_INTP, NPY_INTP};

static void *const data[1] = {NULL};

static int
initialize_ufuncs(PyObject *m)
{
    PyObject *ufunc = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 0,
            PyUFunc_None, "inplace_add", "inplace_add_docstring", 0);
    if (ufunc == NULL) {
        return -1;
    }
    ((PyUFuncObject *)ufunc)->op_flags[0] = NPY_ITER_READWRITE;
    ((PyUFuncObject *)ufunc)->iter_flags = NPY_ITER_REDUCE_OK;
    int result = PyModule_AddObjectRef(m, "inplace_add", ufunc);
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
    .m_name = "_operand_flag_tests",
    .m_size = sizeof(owned_test_state),
    .m_methods = TestMethods,
    .m_slots = owned_test_slots,
    .m_traverse = owned_test_traverse,
    .m_clear = owned_test_clear,
    .m_free = owned_test_free
};
PyMODINIT_FUNC PyInit__operand_flag_tests(void)
{
    return PyModuleDef_Init(&moduledef);
}
