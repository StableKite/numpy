#ifndef NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_

extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;

struct npy_object_reference_plan;

NPY_NO_EXPORT int
_buffer_info_init_native_owner(PyObject *object, void **buffer_info);

NPY_NO_EXPORT PyObject *
_buffer_info_get_native_owner(void *buffer_info);

NPY_NO_EXPORT int
_buffer_info_init_void_object_plan(PyObject *scalar, PyArray_Descr *descr);

NPY_NO_EXPORT int
_buffer_info_init_object_plan(PyArrayObject *array, PyArray_Descr *descr);

NPY_NO_EXPORT const struct npy_object_reference_plan *
_buffer_info_get_object_plan(void *buffer_info);

NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj);

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s);

NPY_NO_EXPORT int
void_getbuffer(PyObject *obj, Py_buffer *view, int flags);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_ */
