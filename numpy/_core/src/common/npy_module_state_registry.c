#include <Python.h>
#include <assert.h>
#include <stdint.h>
#include <stdatomic.h>

#include "npy_module_state_registry.h"

#if PY_VERSION_HEX >= 0x030d0000
typedef struct state_entry {
    int64_t interpreter_id;
    void *state;
    struct state_entry *next;
} state_entry;

/* Only C metadata is shared. No entry owns a Python object or module state. */
static state_entry *entries = NULL;
static PyMutex entries_mutex = {0};
static _Atomic(uint64_t) generation = 1;

#ifdef _MSC_VER
#define NPY_STATE_THREAD_LOCAL __declspec(thread)
#else
#define NPY_STATE_THREAD_LOCAL _Thread_local
#endif

/*
 * The generation changes on every registration/removal, so the cache also
 * handles interpreter-ID reuse after runtime reinitialization. A cached state
 * is used only by an attached thread of its interpreter while its module lives.
 */
static NPY_STATE_THREAD_LOCAL int64_t cached_interpreter_id = -1;
static NPY_STATE_THREAD_LOCAL uint64_t cached_generation = 0;
static NPY_STATE_THREAD_LOCAL void *cached_state = NULL;

NPY_VISIBILITY_HIDDEN int
npy_module_state_register(void *state)
{
    assert(state != NULL);
    int64_t interpreter_id = PyInterpreterState_GetID(PyInterpreterState_Get());
    if (interpreter_id < 0) {
        return -1;
    }
    state_entry *entry = PyMem_RawMalloc(sizeof(*entry));
    if (entry == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    entry->interpreter_id = interpreter_id;
    entry->state = state;

    PyMutex_Lock(&entries_mutex);
    for (state_entry *current = entries; current != NULL;
            current = current->next) {
        if (current->interpreter_id == interpreter_id) {
            PyMutex_Unlock(&entries_mutex);
            PyMem_RawFree(entry);
            PyErr_SetString(PyExc_ImportError,
                    "cannot load module more than once per interpreter");
            return -1;
        }
    }
    entry->next = entries;
    entries = entry;
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    PyMutex_Unlock(&entries_mutex);
    return 0;
}

NPY_VISIBILITY_HIDDEN void *
npy_module_state_lookup(void)
{
    int64_t interpreter_id = PyInterpreterState_GetID(PyInterpreterState_Get());
    uint64_t current_generation = atomic_load_explicit(
            &generation, memory_order_acquire);
    if (cached_interpreter_id == interpreter_id
            && cached_generation == current_generation) {
        return cached_state;
    }

    PyMutex_Lock(&entries_mutex);
    void *state = NULL;
    for (state_entry *current = entries; current != NULL;
            current = current->next) {
        if (current->interpreter_id == interpreter_id) {
            state = current->state;
            break;
        }
    }
    cached_interpreter_id = interpreter_id;
    cached_state = state;
    cached_generation = atomic_load_explicit(&generation, memory_order_relaxed);
    PyMutex_Unlock(&entries_mutex);
    return state;
}

NPY_VISIBILITY_HIDDEN void
npy_module_state_unregister(void *state)
{
    PyMutex_Lock(&entries_mutex);
    state_entry **link = &entries;
    while (*link != NULL) {
        state_entry *entry = *link;
        if (entry->state == state) {
            *link = entry->next;
            atomic_fetch_add_explicit(&generation, 1, memory_order_release);
            PyMutex_Unlock(&entries_mutex);
            PyMem_RawFree(entry);
            return;
        }
        link = &entry->next;
    }
    PyMutex_Unlock(&entries_mutex);
}

NPY_VISIBILITY_HIDDEN size_t
npy_module_state_registry_size(void)
{
    PyMutex_Lock(&entries_mutex);
    size_t count = 0;
    for (state_entry *current = entries; current != NULL;
            current = current->next) {
        count++;
    }
    PyMutex_Unlock(&entries_mutex);
    return count;
}
#endif
