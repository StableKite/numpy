#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_

#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "array_method.h"
#include "dtype_traversal.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "numpy/dtype_api.h"

NPY_NO_EXPORT int
init_dtype_meta_type(PyObject *module);

NPY_NO_EXPORT int
npy_add_dtype_publication(PyObject *module, PyObject *dtype, const char *alias);

/* DType flags, currently private, since we may just expose functions
   Other publicly visible flags are in _dtype_api.h                   */
#define NPY_DT_LEGACY 1 << 0


typedef struct {
    /* DType methods, these could be moved into its own struct */
    PyArrayDTypeMeta_DiscoverDescrFromPyobject *discover_descr_from_pyobject;
    PyArrayDTypeMeta_IsKnownScalarType *is_known_scalar_type;
    PyArrayDTypeMeta_DefaultDescriptor *default_descr;
    PyArrayDTypeMeta_CommonDType *common_dtype;
    PyArrayDTypeMeta_CommonInstance *common_instance;
    PyArrayDTypeMeta_EnsureCanonical *ensure_canonical;
    /*
     * Currently only used for experimental user DTypes.
     */
    PyArrayDTypeMeta_SetItem *setitem;
    PyArrayDTypeMeta_GetItem *getitem;
    /*
     * Either NULL or fetches a clearing function.  Clearing means deallocating
     * any referenced data and setting it to a safe state.  For Python objects
     * this means using `Py_CLEAR` which is equivalent to `Py_DECREF` and
     * setting the `PyObject *` to NULL.
     * After the clear, the data must be fillable via cast/copy and calling
     * clear a second time must be safe.
     * If the DType class does not implement `get_clear_loop` setting
     * NPY_ITEM_REFCOUNT on its dtype instances is invalid.  Note that it is
     * acceptable for  NPY_ITEM_REFCOUNT to indicate references that are not
     * Python objects.
     */
    PyArrayMethod_GetTraverseLoop *get_clear_loop;
    /*
       Either NULL or a function that sets a function pointer to a traversal
       loop that fills an array with zero values appropriate for the dtype. If
       get_fill_zero_loop is undefined or the function pointer set by it is
       NULL, the array buffer is allocated with calloc. If this function is
       defined and it sets a non-NULL function pointer, the array buffer is
       allocated with malloc and the zero-filling loop function pointer is
       called to fill the buffer. For the best performance, avoid using this
       function if a zero-filled array buffer allocated with calloc makes sense
       for the dtype.

       Note that this is currently used only for zero-filling a newly allocated
       array. While it can be used to zero-fill an already-filled buffer, that
       will not work correctly for arrays holding references. If you need to do
       that, clear the array first.
    */
    PyArrayMethod_GetTraverseLoop *get_fill_zero_loop;
    /*
     * Either NULL or a function that performs finalization on a dtype, either
     * returning that dtype or a newly created instance that has the same
     * parameters, if any, as the operand dtype.
     */
    PyArrayDTypeMeta_FinalizeDescriptor *finalize_descr;
    /*
     * Function to fetch constants.  Always defined, but may return "undefined"
     * for all values.
     */
    PyArrayDTypeMeta_GetConstant *get_constant;
    /*
     * The casting implementation (ArrayMethod) to convert between two
     * instances of this DType, stored explicitly for fast access:
     */
    PyArrayMethodObject *within_dtype_castingimpl;
    /*
     * Dictionary of ArrayMethods representing most possible casts
     * (structured and object are exceptions).
     * This should potentially become a weak mapping in the future.
     */
    PyObject *castingimpls;

    /*
     * Storage for `descr->f`, since we may need to allow some customization
     * here at least in a transition period and we need to set it on every
     * dtype instance for backward compatibility.  (Keep this at end)
     */
    PyArray_ArrFuncs f;

    /*
     * Hidden slots for the sort and argsort arraymethods.
     */
    PyArrayMethodObject *sort_meth;
    PyArrayMethodObject *argsort_meth;
    /* Definition for real and imaginary parts, and the (internal) ufuncs */
    PyBoundArrayMethodObject *real_meth;
    PyBoundArrayMethodObject *imag_meth;
} NPY_DType_Slots;

// This must be updated if new slots before within_dtype_castingimpl
// are added
#define NPY_NUM_DTYPE_SLOTS 12
#define NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS 22
#define NPY_DT_MAX_ARRFUNCS_SLOT \
  NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS + _NPY_DT_ARRFUNCS_OFFSET


#define NPY_DT_SLOTS(dtype) ((NPY_DType_Slots *)(dtype)->dt_slots)

static inline int NPY_DT_is_legacy(PyArray_DTypeMeta *dtype) {
    return (dtype->flags & NPY_DT_LEGACY) != 0;
}
static inline int NPY_DT_is_abstract(PyArray_DTypeMeta *dtype) {
    return (dtype->flags & NPY_DT_ABSTRACT) != 0;
}
static inline int NPY_DT_is_parametric(PyArray_DTypeMeta *dtype) {
    return (dtype->flags & NPY_DT_PARAMETRIC) != 0;
}
static inline int NPY_DT_is_numeric(PyArray_DTypeMeta *dtype) {
    return (dtype->flags & NPY_DT_NUMERIC) != 0;
}
static inline int NPY_DT_is_user_defined(PyArray_DTypeMeta *dtype) {
    // New-style user defined dtypes have a type_num of -1 also on DType
    return dtype->type_num == -1;
}
static inline int NPY_DT_has_finalize(PyArray_DTypeMeta *dtype) {
    return NPY_DT_SLOTS(dtype)->finalize_descr != NULL;
}

/*
 * Macros for convenient classmethod calls, since these require
 * the DType both for the slot lookup and as first arguments.
 *
 * (Macros may include NULL checks where appropriate)
 */
#define NPY_DT_CALL_discover_descr_from_pyobject(dtype, obj)  \
    NPY_DT_SLOTS(dtype)->discover_descr_from_pyobject(dtype, obj)
#define NPY_DT_CALL_is_known_scalar_type(dtype, obj)  \
    (NPY_DT_SLOTS(dtype)->is_known_scalar_type != NULL  \
        && NPY_DT_SLOTS(dtype)->is_known_scalar_type(dtype, obj))
#define NPY_DT_CALL_default_descr(dtype)  \
    NPY_DT_SLOTS(dtype)->default_descr(dtype)
#define NPY_DT_CALL_common_dtype(dtype, other)  \
    NPY_DT_SLOTS(dtype)->common_dtype(dtype, other)
#define NPY_DT_CALL_ensure_canonical(descr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->ensure_canonical(descr)
#define NPY_DT_CALL_getitem(descr, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->getitem(descr, data_ptr)
#define NPY_DT_CALL_setitem(descr, value, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->setitem(descr, value, data_ptr)
#define NPY_DT_CALL_get_constant(descr, constant_id, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->get_constant(descr, constant_id, data_ptr)


/*
 * This function will hopefully be phased out or replaced, but was convenient
 * for incremental implementation of new DTypes based on DTypeMeta.
 * (Error checking is not required for DescrFromType, assuming that the
 * type is valid.)
 */
static inline PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}

NPY_NO_EXPORT PyArray_Descr *
dtypemeta_discover_as_default(
        PyArray_DTypeMeta *cls, PyObject* obj);

NPY_NO_EXPORT int
dtypemeta_initialize_struct_from_spec(PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec, int priv);

NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype);

NPY_NO_EXPORT PyArray_DTypeMeta *
dtypemeta_wrap_legacy_descriptor(
    _PyArray_LegacyDescr *descr, PyArray_ArrFuncs *arr_funcs,
    PyTypeObject *dtype_super_class, const char *name, const char *alias);

NPY_NO_EXPORT PyArray_DTypeMeta *
_npy_get_legacy_dtype(int type_num);

/*
 * NumPy's builtin DTypes:
 */

/* Resolve each DType from the module's descriptor table, including aliases. */


#define PyArray_BoolDType (*_npy_get_legacy_dtype(NPY_BOOL))
/* Integers */
#define PyArray_ByteDType (*_npy_get_legacy_dtype(NPY_BYTE))
#define PyArray_UByteDType (*_npy_get_legacy_dtype(NPY_UBYTE))
#define PyArray_ShortDType (*_npy_get_legacy_dtype(NPY_SHORT))
#define PyArray_UShortDType (*_npy_get_legacy_dtype(NPY_USHORT))
#define PyArray_IntDType (*_npy_get_legacy_dtype(NPY_INT))
#define PyArray_UIntDType (*_npy_get_legacy_dtype(NPY_UINT))
#define PyArray_LongDType (*_npy_get_legacy_dtype(NPY_LONG))
#define PyArray_ULongDType (*_npy_get_legacy_dtype(NPY_ULONG))
#define PyArray_LongLongDType (*_npy_get_legacy_dtype(NPY_LONGLONG))
#define PyArray_ULongLongDType (*_npy_get_legacy_dtype(NPY_ULONGLONG))
/* Integer aliases */
#define PyArray_Int8DType (*_npy_get_legacy_dtype(NPY_INT8))
#define PyArray_UInt8DType (*_npy_get_legacy_dtype(NPY_UINT8))
#define PyArray_Int16DType (*_npy_get_legacy_dtype(NPY_INT16))
#define PyArray_UInt16DType (*_npy_get_legacy_dtype(NPY_UINT16))
#define PyArray_Int32DType (*_npy_get_legacy_dtype(NPY_INT32))
#define PyArray_UInt32DType (*_npy_get_legacy_dtype(NPY_UINT32))
#define PyArray_Int64DType (*_npy_get_legacy_dtype(NPY_INT64))
#define PyArray_UInt64DType (*_npy_get_legacy_dtype(NPY_UINT64))
#define PyArray_IntpDType (*_npy_get_legacy_dtype(NPY_INTP))
#define PyArray_UIntpDType (*_npy_get_legacy_dtype(NPY_UINTP))
#define PyArray_DefaultIntDType (*_npy_get_legacy_dtype(NPY_DEFAULT_INT))
/* Floats */
#define PyArray_HalfDType (*_npy_get_legacy_dtype(NPY_HALF))
#define PyArray_FloatDType (*_npy_get_legacy_dtype(NPY_FLOAT))
#define PyArray_DoubleDType (*_npy_get_legacy_dtype(NPY_DOUBLE))
#define PyArray_LongDoubleDType (*_npy_get_legacy_dtype(NPY_LONGDOUBLE))
/* Complex */
#define PyArray_CFloatDType (*_npy_get_legacy_dtype(NPY_CFLOAT))
#define PyArray_CDoubleDType (*_npy_get_legacy_dtype(NPY_CDOUBLE))
#define PyArray_CLongDoubleDType (*_npy_get_legacy_dtype(NPY_CLONGDOUBLE))
/* String/Bytes */
#define PyArray_BytesDType (*_npy_get_legacy_dtype(NPY_STRING))
#define PyArray_UnicodeDType (*_npy_get_legacy_dtype(NPY_UNICODE))
NPY_NO_EXPORT PyArray_DTypeMeta *
_npy_get_string_dtype(void);
#define PyArray_StringDType (*_npy_get_string_dtype())
/* Datetime/Timedelta */
#define PyArray_DatetimeDType (*_npy_get_legacy_dtype(NPY_DATETIME))
#define PyArray_TimedeltaDType (*_npy_get_legacy_dtype(NPY_TIMEDELTA))
/* Object/Void */
#define PyArray_ObjectDType (*_npy_get_legacy_dtype(NPY_OBJECT))
#define PyArray_VoidDType (*_npy_get_legacy_dtype(NPY_VOID))

#ifdef __cplusplus
}
#endif


/* Internal version see dtypmeta.c for more information. */
static inline PyArray_ArrFuncs *
PyDataType_GetArrFuncs(const PyArray_Descr *descr)
{
    return &NPY_DT_SLOTS(NPY_DTYPE(descr))->f;
}

/*
 * Internal versions.  Note that `PyArray_Pack` or `PyArray_Scalar` are often
 * preferred (PyArray_Pack knows how to cast and deal with arrays,
 * PyArray_Scalar will convert to the Python type).
 */
static inline PyObject *
PyArray_GETITEM(const PyArrayObject *arr, const char *itemptr)
{
    return PyDataType_GetArrFuncs(PyArray_DESCR(arr))->getitem(
            (void *)itemptr, (PyArrayObject *)arr);
}

static inline int
PyArray_SETITEM(PyArrayObject *arr, char *itemptr, PyObject *v)
{
    return NPY_DT_CALL_setitem(PyArray_DESCR(arr), v, itemptr);
}

// Like PyArray_DESCR_REPLACE, but calls ensure_canonical instead of DescrNew
#define PyArray_DESCR_REPLACE_CANONICAL(descr) do { \
                PyArray_Descr *_new_ = NPY_DT_CALL_ensure_canonical(descr); \
                Py_XSETREF(descr, _new_);  \
        } while(0)


// Get the pointer to the PyArray_DTypeMeta for the type associated with the typenum.
static inline PyArray_DTypeMeta *
typenum_to_dtypemeta(enum NPY_TYPES typenum) {
    PyArray_Descr * descr = PyArray_DescrFromType(typenum);
    Py_DECREF(descr);
    return NPY_DTYPE(descr);
}


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_ */
