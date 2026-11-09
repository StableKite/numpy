/* Included by the validated generated-code adapter, only for CPython 3.15+.
 * Detached regions retain a borrowed caller solely for their dynamic extent.
 * No PyGILState attachment to the main interpreter is used. */
#include "numpy/interpreter_context.h"

static NPY_SI_Context *npy_random_module_context(void);
static int *npy_random_module_runtime_version(void);
static void *npy_random_attached_module_state(void);
#define NPY_API_MODULE_CONTEXT (npy_random_module_context())
#define NPY_API_MODULE_RUNTIME_VERSION (*npy_random_module_runtime_version())

/* Late deallocation may follow module m_clear. The object's real type chain
 * retains the native base layout; an emptied Cython cache does not. */
static PyTypeObject *
npy_random_next_dealloc_base(PyObject *object, destructor current)
{
    PyTypeObject *type = Py_TYPE(object);
    while (type != NULL && type->tp_dealloc != current) type = type->tp_base;
    while (type != NULL && type->tp_dealloc == current) type = type->tp_base;
    return type;
}

#if defined(_MSC_VER)
static __declspec(thread) PyThreadState *npy_random_detached_owner;
static __declspec(thread) void *npy_random_detached_state;
#else
static _Thread_local PyThreadState *npy_random_detached_owner;
static _Thread_local void *npy_random_detached_state;
#endif

typedef struct {
    PyThreadState *owner;
    void *state;
} npy_random_detached_scope;

static PyThreadState *
npy_random_save_thread(npy_random_detached_scope *previous)
{
    previous->owner = npy_random_detached_owner;
    previous->state = npy_random_detached_state;
    npy_random_detached_owner = PyThreadState_Get();
    /* Header inline functions may inspect the immutable API runtime version
     * inside a numeric region. Capture its native module allocation attached. */
    npy_random_detached_state = npy_random_attached_module_state();
    return PyEval_SaveThread();
}

static void
npy_random_restore_thread(PyThreadState *caller, npy_random_detached_scope previous)
{
    PyEval_RestoreThread(caller);
    npy_random_detached_owner = previous.owner;
    npy_random_detached_state = previous.state;
}

typedef PyThreadState *npy_random_attachment;

static npy_random_attachment
npy_random_ensure_attached(void)
{
    if (PyThreadState_GetUnchecked() != NULL) {
        return NULL;
    }
    PyThreadState *owner = npy_random_detached_owner;
    if (owner == NULL) {
        Py_FatalError("NumPy random: native callback has no owning interpreter");
    }
    PyEval_RestoreThread(owner);
    return owner;
}

static void
npy_random_release_attached(npy_random_attachment attachment)
{
    if (attachment != NULL) {
        PyThreadState *owner = PyEval_SaveThread();
        if (owner != attachment) {
            Py_FatalError("NumPy random: native callback changed its interpreter");
        }
    }
}

/* All generated acquisition sites pair this token with Release, including
 * Cython's error paths in nogil buffer helpers. Python.h has already loaded. */
#define PyGILState_STATE npy_random_attachment
#define PyGILState_Ensure npy_random_ensure_attached
#define PyGILState_Release npy_random_release_attached
