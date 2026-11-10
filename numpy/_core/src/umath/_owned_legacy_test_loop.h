#ifndef NUMPY_OWNED_LEGACY_TEST_LOOP_H
#define NUMPY_OWNED_LEGACY_TEST_LOOP_H
#include "array_method.h"

/* Private to the built-in test extension. Keep the original legacy selector,
 * flags, loop, data, and return convention. Only add a call-scoped owner.
 * No Python object or thread state is retained beyond the active invocation. */
static _Thread_local PyThreadState *owned_legacy_test_caller;
typedef struct {
    NpyAuxData base;
    NpyAuxData *inner;
    PyArrayMethod_StridedLoop *loop;
    PyThreadState *owner;
} owned_legacy_test_aux;

static void
owned_legacy_test_free(NpyAuxData *data)
{
    owned_legacy_test_aux *aux = (owned_legacy_test_aux *)data;
    NPY_AUXDATA_FREE(aux->inner);
    PyMem_Free(aux);
}
static int
owned_legacy_test_loop(PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides, NpyAuxData *data)
{
    owned_legacy_test_aux *aux = (owned_legacy_test_aux *)data;
    PyThreadState *previous = owned_legacy_test_caller;
    owned_legacy_test_caller = aux->owner;
    int result = aux->loop(context, args, dimensions, strides, aux->inner);
    owned_legacy_test_caller = previous;
    return result;
}
static int
owned_legacy_test_get_loop(PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_data,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArrayMethod_GetLoop *original =
            (PyArrayMethod_GetLoop *)context->method->static_data;
    owned_legacy_test_aux *aux = PyMem_Calloc(1, sizeof(*aux));
    if (aux == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    if (original(context, aligned, move_references, strides,
            &aux->loop, &aux->inner, flags) < 0) {
        PyMem_Free(aux);
        return -1;
    }
    aux->base.free = owned_legacy_test_free;
    aux->base.clone = NULL;  /* original legacy auxdata is not cloneable */
    aux->owner = PyThreadState_Get();
    *out_loop = owned_legacy_test_loop;
    *out_data = (NpyAuxData *)aux;
    return 0;
}
static int
owned_legacy_test_install(PyUFuncObject *ufunc)
{
    Py_ssize_t position = 0;
    PyObject *key, *value;
    /* Called only while this newly created, unexported ufunc is private. */
    while (PyDict_Next(ufunc->_loops, &position, &key, &value)) {
        PyArrayMethodObject *method =
                (PyArrayMethodObject *)PyTuple_GET_ITEM(value, 1);
        if (method->static_data != NULL || method->get_strided_loop == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                    "unexpected legacy test loop ownership");
            return -1;
        }
        method->static_data = (void *)method->get_strided_loop;
        method->get_strided_loop = owned_legacy_test_get_loop;
    }
    return 0;
}
#endif
