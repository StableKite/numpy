#ifndef NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_

NPY_NO_EXPORT int
init_nditer_type(PyObject *module);

NPY_NO_EXPORT PyObject *
NpyIter_NestedIters(PyObject *NPY_UNUSED(self),
                    PyObject *args, PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_ */
