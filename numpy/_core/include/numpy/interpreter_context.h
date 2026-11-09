#ifndef NUMPY_CORE_INCLUDE_NUMPY_INTERPRETER_CONTEXT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_INTERPRETER_CONTEXT_H_

/* Explicit per-module API ownership. This header does not initialize or cache
 * the legacy process-wide PyArray_API/PyUFunc_API variables. */
#include <Python.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NPY_SI_CONTEXT_ABI_VERSION 1
#define NPY_SI_CONTEXT_CAPSULE "numpy._core._multiarray_umath._SI_CONTEXT_API_v1"

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    /* New reference; the module must be the actual native NumPy module. */
    PyObject *(*get_owner)(PyObject *module);
    int (*check_owner)(PyObject *owner);
    /* Tables and object/type getters borrow from the owned context. */
    void **(*get_array_api)(PyObject *owner);
    void **(*get_ufunc_api)(PyObject *owner);
    PyObject *(*get_bool)(PyObject *owner, int value);
    PyTypeObject *(*get_scalar_type)(PyObject *owner, int type_num);
} NPY_SI_ContextAPI;

typedef struct {
    const NPY_SI_ContextAPI *api;
    PyObject *owner;
    void **array_api;
    void **ufunc_api;
    PyObject *bool_values[2];  /* Borrowed; never indexed as contiguous cells. */
} NPY_SI_Context;

static inline int
NPY_SI_Context_Traverse(NPY_SI_Context *context, visitproc visit, void *arg)
{
    Py_VISIT(context->owner);
    return 0;
}

static inline void
NPY_SI_Context_Clear(NPY_SI_Context *context)
{
    PyObject *owner = context->owner;
    context->owner = NULL;
    context->api = NULL;
    context->array_api = NULL;
    context->ufunc_api = NULL;
    context->bool_values[0] = NULL;
    context->bool_values[1] = NULL;
    Py_XDECREF(owner);
}

/* A zero-initialized context belongs in the importing extension's module
 * state. Traverse it in m_traverse and clear it in m_clear/m_free. Calls that
 * use its Python objects/API must run in its owning interpreter, attached to
 * Python, just like ordinary Python C API calls. */
static inline int
NPY_SI_Context_Import(NPY_SI_Context *context)
{
    if (context->owner != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NumPy interpreter context is already initialized");
        return -1;
    }
    PyObject *module = PyImport_ImportModule("numpy._core._multiarray_umath");
    if (module == NULL) {
        return -1;
    }
    PyObject *capsule = PyObject_GetAttrString(module, "_SI_CONTEXT_API_v1");
    if (capsule == NULL) {
        Py_DECREF(module);
        return -1;
    }
    const NPY_SI_ContextAPI *api = (const NPY_SI_ContextAPI *)
            PyCapsule_GetPointer(capsule, NPY_SI_CONTEXT_CAPSULE);
    Py_DECREF(capsule);
    if (api == NULL) {
        Py_DECREF(module);
        return -1;
    }
    if (api->abi_version != NPY_SI_CONTEXT_ABI_VERSION
            || api->struct_size < sizeof(NPY_SI_ContextAPI)) {
        Py_DECREF(module);
        PyErr_SetString(PyExc_ImportError, "incompatible NumPy interpreter context API");
        return -1;
    }
    PyObject *owner = api->get_owner(module);
    Py_DECREF(module);
    if (owner == NULL) {
        return -1;
    }
    context->api = api;
    context->owner = owner;
    context->array_api = api->get_array_api(owner);
    if (context->array_api == NULL) {
        goto fail;
    }
    context->ufunc_api = api->get_ufunc_api(owner);
    if (context->ufunc_api == NULL) {
        goto fail;
    }
    context->bool_values[0] = api->get_bool(owner, 0);
    if (context->bool_values[0] == NULL) {
        goto fail;
    }
    context->bool_values[1] = api->get_bool(owner, 1);
    if (context->bool_values[1] == NULL) {
        goto fail;
    }
    return 0;
fail:
    NPY_SI_Context_Clear(context);
    return -1;
}

static inline int
NPY_SI_Context_Check(NPY_SI_Context *context)
{
    if (context->owner == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NumPy interpreter context is not initialized");
        return -1;
    }
    return context->api->check_owner(context->owner);
}

static inline PyObject *
NPY_SI_ScalarNew(NPY_SI_Context *context, int type_num)
{
    if (NPY_SI_Context_Check(context) < 0) {
        return NULL;
    }
    PyTypeObject *type = context->api->get_scalar_type(context->owner, type_num);
    /* Run the real constructor, including CPython str/bytes initialization. */
    return type == NULL ? NULL : PyObject_CallNoArgs((PyObject *)type);
}

static inline PyObject *
NPY_SI_ScalarCall(NPY_SI_Context *context, int type_num, PyObject *args, PyObject *kwargs)
{
    if (NPY_SI_Context_Check(context) < 0) {
        return NULL;
    }
    PyTypeObject *type = context->api->get_scalar_type(context->owner, type_num);
    return type == NULL ? NULL : PyObject_Call((PyObject *)type, args, kwargs);
}

#ifdef __cplusplus
}
#endif
#endif  /* NUMPY_CORE_INCLUDE_NUMPY_INTERPRETER_CONTEXT_H_ */
