#ifndef NUMPY_CORE_SRC_COMMON_NPY_MODULE_STATE_REGISTRY_H_
#define NUMPY_CORE_SRC_COMMON_NPY_MODULE_STATE_REGISTRY_H_

#include <Python.h>
#include "numpy/npy_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#if PY_VERSION_HEX >= 0x030d0000
/*
 * A non-owning lookup for legacy C entry points without a module/type argument.
 * The module owns its state and must unregister it before freeing it. Prefer
 * PyModule_GetState or PyType_GetModuleState when an owner is available.
 */
NPY_VISIBILITY_HIDDEN int
npy_module_state_register(void *state);

NPY_VISIBILITY_HIDDEN void *
npy_module_state_lookup(void);

NPY_VISIBILITY_HIDDEN void
npy_module_state_unregister(void *state);

NPY_VISIBILITY_HIDDEN size_t
npy_module_state_registry_size(void);
#endif

#ifdef __cplusplus
}
#endif
#endif
