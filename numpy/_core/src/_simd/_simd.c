#include "_simd.h"

#include "numpy/npy_math.h"

static PyObject *
get_floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    return PyLong_FromLong(npy_get_floatstatus());
}

static PyObject *
clear_floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    npy_clear_floatstatus();
    Py_RETURN_NONE;
}

static PyMethodDef _simd_methods[] = {
    {"get_floatstatus", get_floatstatus, METH_NOARGS, NULL},
    {"clear_floatstatus", clear_floatstatus, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static int
simd_exec(PyObject *m)
{
    if (npy_cpu_init() < 0) {
        return -1;
    }
    PyObject *targets = PyDict_New();
    if (targets == NULL) {
        goto err;
    }
    if (PyModule_AddObject(m, "targets", targets) < 0) {
        Py_DECREF(targets);
        goto err;
    }
    // add keys for non-supported optimizations with None value
    #define ATTACH_MODULE(TESTED_FEATURES, TARGET_NAME, MAKE_MSVC_HAPPY)       \
        {                                                                      \
            PyObject *simd_mod;                                                \
            if (!TESTED_FEATURES) {                                            \
                Py_INCREF(Py_None);                                            \
                simd_mod = Py_None;                                            \
            } else {                                                           \
                simd_mod = NPY_CAT(simd_create_module_, TARGET_NAME)();        \
                if (simd_mod == NULL) {                                        \
                    goto err;                                                  \
                }                                                              \
            }                                                                  \
            const char *target_name = NPY_TOSTRING(TARGET_NAME);               \
            if (PyDict_SetItemString(targets, target_name, simd_mod) < 0) {    \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
                                               \
            if (PyModule_AddObjectRef(m, target_name, simd_mod) < 0) {            \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
            Py_DECREF(simd_mod);                                               \
        }

    #define ATTACH_BASELINE_MODULE(MAKE_MSVC_HAPPY)                            \
        {                                                                      \
            PyObject *simd_mod = simd_create_module();                         \
            if (simd_mod == NULL) {                                            \
                goto err;                                                      \
            }                                                                  \
            if (PyDict_SetItemString(targets, "baseline", simd_mod) < 0) {     \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
                                               \
            if (PyModule_AddObjectRef(m, "baseline", simd_mod) < 0) {             \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
            Py_DECREF(simd_mod);                                               \
        }
    NPY_MTARGETS_CONF_DISPATCH(NPY_CPU_HAVE, ATTACH_MODULE, MAKE_MSVC_HAPPY)
    NPY_MTARGETS_CONF_BASELINE(ATTACH_BASELINE_MODULE, MAKE_MSVC_HAPPY)

    return 0;
err:
    return -1;
}

static PyModuleDef_Slot simd_slots[] = {
    {Py_mod_exec, simd_exec},
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
static struct PyModuleDef simd_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "numpy._core._simd",
    .m_size = 0,
    .m_methods = _simd_methods,
    .m_slots = simd_slots
};
PyMODINIT_FUNC PyInit__simd(void)
{
    return PyModuleDef_Init(&simd_def);
}
