#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_

/*
 * Field lists for the PyObject members of multiarray_umath_state, used to
 * expand the module traverse and clear functions.
 *
 * interned_str.errmode_strings is an array, so it is handled separately at
 * each use site.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define NPY_INTERNED_STR_FIELDS(F) \
    F(current_allocator) \
    F(array) \
    F(array_function) \
    F(array_struct) \
    F(array_priority) \
    F(array_interface) \
    F(array_wrap) \
    F(array_finalize) \
    F(array_ufunc) \
    F(numpy_dtype) \
    F(implementation) \
    F(axis1) \
    F(axis2) \
    F(item) \
    F(like) \
    F(numpy) \
    F(where) \
    F(convert) \
    F(preserve) \
    F(convert_if_no_array) \
    F(cpu) \
    F(dtype) \
    F(array_err_msg_substr) \
    F(out) \
    F(__dlpack__) \
    F(pyvals_name) \
    F(legacy) \
    F(__doc__) \
    F(__signature__) \
    F(copy) \
    F(dl_device) \
    F(max_version) \
    F(array_dealloc) \
    F(real) \
    F(imag) \
    F(sort) \
    F(argsort) \
    F(_set_dtype)

#define NPY_STATIC_PYDATA_FIELDS(F) \
    F(default_truediv_type_tup) \
    F(default_extobj_capsule) \
    F(npy_extobj_contextvar) \
    F(ndarray_array_ufunc) \
    F(ndarray_array_finalize) \
    F(ndarray_array_function) \
    F(ndarray_set_dtype) \
    F(ndarray_dtype_descr) \
    F(one_obj) \
    F(zero_obj) \
    F(zero_pyint_like_arr) \
    F(AxisError) \
    F(ComplexWarning) \
    F(DTypePromotionError) \
    F(TooHardError) \
    F(VisibleDeprecationWarning) \
    F(_CopyMode) \
    F(_NoValue) \
    F(_ArrayMemoryError) \
    F(_UFuncBinaryResolutionError) \
    F(_UFuncInputCastingError) \
    F(_UFuncNoLoopError) \
    F(_UFuncOutputCastingError) \
    F(math_floor_func) \
    F(math_ceil_func) \
    F(math_trunc_func) \
    F(math_gcd_func) \
    F(os_PathLike) \
    F(os_fspath) \
    F(format_options) \
    F(kwnames_is_copy) \
    F(axes_1d_obj_kwargs) \
    F(axes_2d_obj_kwargs) \
    F(cpu_dispatch_registry) \
    F(VoidToGenericMethod) \
    F(GenericToVoidMethod) \
    F(ObjectToGenericMethod) \
    F(GenericToObjectMethod) \
    F(dl_call_kwnames) \
    F(dl_cpu_device_tuple) \
    F(dl_max_version) \
    F(dlpack_dtype_registry) \
    F(dlpack_export_registry)

#define NPY_RUNTIME_IMPORTS_FIELDS(F) \
    F(_add_dtype_helper) \
    F(_all) \
    F(_amax) \
    F(_amin) \
    F(_any) \
    F(array_function_errmsg_formatter) \
    F(array_ufunc_errmsg_formatter) \
    F(_clip) \
    F(_commastring) \
    F(_convert_to_stringdtype_kwargs) \
    F(_default_array_repr) \
    F(_default_array_str) \
    F(_dump) \
    F(_dumps) \
    F(_getfield_is_safe) \
    F(internal_gcd_func) \
    F(_mean) \
    F(NO_NEP50_WARNING) \
    F(npy_ctypes_check) \
    F(numpy_matrix) \
    F(_prod) \
    F(_promote_fields) \
    F(_std) \
    F(_sum) \
    F(_ufunc_doc_signature_formatter) \
    F(_ufunc_inspect_signature_builder) \
    F(_usefields) \
    F(_var) \
    F(_view_is_safe) \
    F(_void_scalar_to_string)

#define NPY_N_OPS_FIELDS(F) \
    F(add) \
    F(subtract) \
    F(multiply) \
    F(divide) \
    F(remainder) \
    F(divmod) \
    F(power) \
    F(square) \
    F(reciprocal) \
    F(_ones_like) \
    F(sqrt) \
    F(cbrt) \
    F(negative) \
    F(positive) \
    F(absolute) \
    F(invert) \
    F(left_shift) \
    F(right_shift) \
    F(bitwise_and) \
    F(bitwise_xor) \
    F(bitwise_or) \
    F(less) \
    F(less_equal) \
    F(equal) \
    F(not_equal) \
    F(greater) \
    F(greater_equal) \
    F(floor_divide) \
    F(true_divide) \
    F(logical_or) \
    F(logical_and) \
    F(floor) \
    F(ceil) \
    F(maximum) \
    F(minimum) \
    F(rint) \
    F(conjugate) \
    F(matmul) \
    F(clip) \
    F(real) \
    F(imag)

#define NPY_MODULE_STATE_OBJECT_FIELDS(F) \
    F(typeDict)        \
    F(bool_false)                     \
    F(bool_true)                      \
    F(current_handler) \
    F(default_mem_handler) \
    F(global_pytype_to_type_dict)

/*
 * Heap types created with PyType_FromModuleAndSpec or PyType_FromMetaclass
 * during module execution.  Each field owns its reference.
 */
#define NPY_MODULE_STATE_TYPE_FIELDS(F) \
    F(_dtype_meta_type)               \
    F(descr_type)                     \
    F(string_dtype_type)              \
    F(array_method_type)              \
    F(bound_array_method_type)        \
    F(ufunc_type)                     \
    F(arrayiter_type)                 \
    F(arraymultiiter_type)            \
    F(neighborhood_iter_type)         \
    F(mapiter_type)                   \
    F(nditer_type)                    \
    F(array_type)                     \
    F(generic_scalar_type)            \
    F(number_scalar_type)             \
    F(integer_scalar_type)            \
    F(signedinteger_scalar_type)      \
    F(unsignedinteger_scalar_type)    \
    F(inexact_scalar_type)            \
    F(floating_scalar_type)           \
    F(complexfloating_scalar_type)    \
    F(flexible_scalar_type)           \
    F(character_scalar_type)          \
    F(void_scalar_type)               \
    F(string_scalar_type)             \
    F(unicode_scalar_type)            \
    F(datetime_scalar_type)           \
    F(timedelta_scalar_type)          \
    F(byte_scalar_type)               \
    F(short_scalar_type)              \
    F(int_scalar_type)                \
    F(long_scalar_type)               \
    F(longlong_scalar_type)           \
    F(ubyte_scalar_type)              \
    F(ushort_scalar_type)             \
    F(uint_scalar_type)               \
    F(ulong_scalar_type)              \
    F(ulonglong_scalar_type)          \
    F(half_scalar_type)               \
    F(float_scalar_type)              \
    F(double_scalar_type)             \
    F(longdouble_scalar_type)         \
    F(cfloat_scalar_type)             \
    F(cdouble_scalar_type)            \
    F(clongdouble_scalar_type)        \
    F(object_scalar_type)             \
    F(bool_scalar_type)               \
    F(integer_abstract_dtype)         \
    F(float_abstract_dtype)           \
    F(complex_abstract_dtype)         \
    F(pylong_dtype)                   \
    F(pyfloat_dtype)                  \
    F(pycomplex_dtype)                \
    F(PyArrayFlags_Type)              \
    F(PyArrayArrayConverter_Type)     \
    F(PyArrayFunctionDispatcher_Type) \
    F(NpyBusDayCalendar_Type)

/*
 * Expand a field list into the PyObject * members it names, so the structs
 * below are declared from the same list the traverse and clear functions use.
 */
#define NPY_DECLARE_ONE_PYOBJECT_FIELD(name) PyObject *name;
#define NPY_DECLARE_PYOBJECT_FIELDS(list) list(NPY_DECLARE_ONE_PYOBJECT_FIELD)

/* Types have the same ownership rules, with their actual C pointer type. */
#define NPY_DECLARE_ONE_TYPE_FIELD(name) PyTypeObject *name;
#define NPY_DECLARE_TYPE_FIELDS(list) list(NPY_DECLARE_ONE_TYPE_FIELD)

#define NPY_FIELD_COUNT_ONE(name) + 1
#define NPY_FIELD_COUNT(list) (0 list(NPY_FIELD_COUNT_ONE))

#define NPY_ERRMODE_STRING_COUNT 6

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_FIELDS_H_ */
