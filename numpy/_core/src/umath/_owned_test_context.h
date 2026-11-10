#ifndef NUMPY_OWNED_TEST_CONTEXT_H
#define NUMPY_OWNED_TEST_CONTEXT_H
#include "numpy/interpreter_context.h"
typedef struct {
    NPY_SI_Context numpy;
    int runtime_version;
} owned_test_state;
static _Thread_local owned_test_state *owned_test_active;
#define NPY_API_MODULE_CONTEXT (&owned_test_active->numpy)
#define NPY_API_MODULE_RUNTIME_VERSION (owned_test_active->runtime_version)

static int
owned_test_traverse(PyObject *module, visitproc visit, void *arg)
{
    owned_test_state *state = PyModule_GetState(module);
    return state == NULL ? 0 : NPY_SI_Context_Traverse(&state->numpy, visit, arg);
}
static int
owned_test_clear(PyObject *module)
{
    owned_test_state *state = PyModule_GetState(module);
    if (state != NULL) {
        NPY_SI_Context_Clear(&state->numpy);
    }
    return 0;
}
static void
owned_test_free(void *module)
{
    owned_test_clear((PyObject *)module);
}
static int
owned_test_initialize(PyObject *module)
{
    owned_test_state *state = PyModule_GetState(module);
    if (state->numpy.owner == NULL && NPY_SI_Context_Import(&state->numpy) < 0) {
        return -1;
    }
    typedef unsigned int (*feature_version_func)(void);
    state->runtime_version = (int)((feature_version_func)state->numpy.array_api[211])();
    return 0;
}
#endif
