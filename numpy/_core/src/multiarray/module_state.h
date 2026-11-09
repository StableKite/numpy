#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_

#include <stddef.h>  /* for offsetof */

#include "npy_static_data.h"
#include "npy_import.h"
#include "multiarraymodule.h"
#include "number.h"
#include "module_state_fields.h"
#include "npy_module_state_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    PyTypeObject *type;
    int typenum;
} npy_scalar_type_entry;

typedef struct multiarray_umath_state {
    npy_interned_str_struct    interned_str;
    npy_static_pydata_struct   static_pydata;
    npy_static_cdata_struct    static_cdata;
    npy_runtime_imports_struct runtime_imports;
    npy_global_state_struct    global_state;

    NPY_DECLARE_PYOBJECT_FIELDS(NPY_MODULE_STATE_OBJECT_FIELDS)
    NPY_DECLARE_TYPE_FIELDS(NPY_MODULE_STATE_TYPE_FIELDS)
    NumericOps n_ops;
    _PyArray_LegacyDescr *builtin_descrs[NPY_NTYPES_LEGACY];
    /* Borrowed object/function addresses; the module owns the table buffers. */
    void **array_api;
    void **ufunc_api;
    _PyArray_LegacyDescr **user_descrs;
    int num_user_types;
    NPY_CASTING default_assign_casting;
    /* Borrowed class identities, without additional Python references. */
    npy_scalar_type_entry scalar_type_lookup[NPY_NTYPES_LEGACY];
    /* Native protocol storage; contains function pointers, no PyObjects. */
    PyNumberMethods scalar_math_protocols[NPY_NTYPES_LEGACY];
    PyNumberMethods generic_scalar_protocol;
    PyNumberMethods bool_scalar_protocol;
    struct npy_bool_scalar_pool *bool_scalar_pool;
    /* Main uses embedded state. A child module slot owns lifetime_owner and
     * borrows indirect_state. In the payload lifetime_owner borrows self. */
    PyObject *lifetime_owner;
    struct multiarray_umath_state *indirect_state;
} multiarray_umath_state;

/* The structs are generated from the lists, so these only catch a stray
 * member added by hand outside them. */
static_assert(sizeof(npy_interned_str_struct) ==
        (NPY_FIELD_COUNT(NPY_INTERNED_STR_FIELDS) + NPY_ERRMODE_STRING_COUNT)
                * sizeof(PyObject *),
        "npy_interned_str_struct member missing from NPY_INTERNED_STR_FIELDS");

static_assert(sizeof(npy_static_pydata_struct) ==
        NPY_FIELD_COUNT(NPY_STATIC_PYDATA_FIELDS) * sizeof(PyObject *),
        "npy_static_pydata_struct member missing from "
        "NPY_STATIC_PYDATA_FIELDS");

static_assert(sizeof(npy_runtime_imports_struct) ==
        NPY_FIELD_COUNT(NPY_RUNTIME_IMPORTS_FIELDS) * sizeof(PyObject *),
        "npy_runtime_imports_struct member missing from "
        "NPY_RUNTIME_IMPORTS_FIELDS");

static_assert(sizeof(NumericOps) ==
        NPY_FIELD_COUNT(NPY_N_OPS_FIELDS) * sizeof(PyObject *),
        "NumericOps member missing from NPY_N_OPS_FIELDS");

/* The loose members sit contiguously between the sub-structs and n_ops. */
static_assert(offsetof(multiarray_umath_state, n_ops) -
        offsetof(multiarray_umath_state, typeDict) ==
        (NPY_FIELD_COUNT(NPY_MODULE_STATE_OBJECT_FIELDS) +
         NPY_FIELD_COUNT(NPY_MODULE_STATE_TYPE_FIELDS)) * sizeof(PyObject *),
        "multiarray_umath_state member missing from "
        "NPY_MODULE_STATE_OBJECT_FIELDS or NPY_MODULE_STATE_TYPE_FIELDS");

static inline multiarray_umath_state *
get_module_state(PyObject *module)
{
    void *state = PyModule_GetState(module);
    assert(state != NULL);
    multiarray_umath_state *slot = (multiarray_umath_state *)state;
    return slot->indirect_state == NULL ? slot : slot->indirect_state;
}

/* Prefer an explicit module/type owner. This fallback requires an attached
 * thread state, so loops that detach must capture their state beforehand. */
#if PY_VERSION_HEX >= 0x030d0000
static inline multiarray_umath_state *
_npy_get_current_module_state(void)
{
    multiarray_umath_state *state =
            (multiarray_umath_state *)npy_module_state_lookup();
    assert(state != NULL);
    return state;
}
#define _npy_module_state _npy_get_current_module_state()
#else
/* Older builds retain their existing single-interpreter behavior. */
NPY_VISIBILITY_HIDDEN extern multiarray_umath_state *_npy_module_state;
#endif

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_ */
