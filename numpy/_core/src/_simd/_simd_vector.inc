/**
 * This file is included by `_simd.dispatch.c.src`. Its contents are affected by the simd configuration, and
 * therefore must be built multiple times. Making it a standalone `.c` file with `NPY_VISIBILITY_HIDDEN`
 * symbols would require judicious use of `NPY_CPU_DISPATCH_DECLARE` and `NPY_CPU_DISPATCH_CURFX`, which was
 * deemed too harmful to readability.
 */
/************************************
 ** Private Definitions
 ************************************/
static Py_ssize_t
simd__vector_length(PySIMDVectorObject *self)
{
    return simd_data_getinfo(self->dtype)->nlanes;
}
static PyObject *
simd__vector_item(PySIMDVectorObject *self, Py_ssize_t i)
{
    const simd_data_info *info = simd_data_getinfo(self->dtype);
    int nlanes = info->nlanes;
    if (i >= nlanes) {
        PyErr_SetString(PyExc_IndexError, "vector index out of range");
        return NULL;
    }
    npyv_lanetype_u8 *src = self->data + i * info->lane_size;
    simd_data data;
    memcpy(&data.u64, src, info->lane_size);
    return simd_scalar_to_number(data, info->to_scalar);
}

static PyObject *
simd__vector_name(PySIMDVectorObject *self, void *NPY_UNUSED(ignored))
{
    return PyUnicode_FromString(simd_data_getinfo(self->dtype)->pyname);
}
static PyGetSetDef simd__vector_getset[] = {
    { "__name__", (getter)simd__vector_name, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyObject *
simd__vector_repr(PySIMDVectorObject *self)
{
    PyObject *obj = PySequence_List((PyObject*)self);
    if (obj != NULL) {
        const char *type_name = simd_data_getinfo(self->dtype)->pyname;
        PyObject *repr = PyUnicode_FromFormat("<%s of %R>", type_name, obj);
        Py_DECREF(obj);
        return repr;
    }
    return obj;
}
static PyObject *
simd__vector_compare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *obj;
    if (PyTuple_Check(other)) {
        obj = PySequence_Tuple(self);
    } else if (PyList_Check(other)) {
        obj = PySequence_List(self);
    } else {
        obj = PySequence_Fast(self, "invalid argument, expected a vector");
    }
    if (obj != NULL) {
        PyObject *rich = PyObject_RichCompare(obj, other, cmp_op);
        Py_DECREF(obj);
        return rich;
    }
    return obj;
}
static int
simd__vector_traverse(PySIMDVectorObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}
static void
simd__vector_dealloc(PySIMDVectorObject *self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *type = Py_TYPE(self);
    PyObject_GC_Del(self);
    Py_DECREF(type);
}
static PyObject *
simd__vector_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyErr_SetString(PyExc_TypeError, "SIMD vectors must be created by target intrinsics");
    return NULL;
}
static PyType_Slot simd__vector_slots[] = {
    {Py_tp_dealloc, simd__vector_dealloc},
    {Py_tp_traverse, simd__vector_traverse},
    {Py_tp_new, simd__vector_new},
    {Py_tp_repr, simd__vector_repr},
    {Py_tp_richcompare, simd__vector_compare},
    {Py_tp_getset, simd__vector_getset},
    {Py_sq_length, simd__vector_length},
    {Py_sq_item, simd__vector_item},
    {0, NULL}
};
static PyType_Spec simd__vector_spec = {
    .name = "numpy._core._simd." NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(VECTOR)),
    .basicsize = sizeof(PySIMDVectorObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots = simd__vector_slots
};

/************************************
 ** Protected Definitions
 ************************************/
/*
 * Force inlining the following functions on CYGWIN to avoid spilling vector
 * registers into the stack to workaround GCC/WIN64 bug that performs
 * miss-align load variable of 256/512-bit vector from non-aligned
 * 256/512-bit stack pointer.
 *
 * check the following links for more clarification:
 * https://github.com/numpy/numpy/pull/18330#issuecomment-821539919
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49001
 */
#if defined(__CYGWIN__) || (defined(__GNUC__) && defined(_WIN64))
    #define CYG_FINLINE NPY_FINLINE
#else
    #define CYG_FINLINE static
#endif
CYG_FINLINE PySIMDVectorObject *
PySIMDVector_FromData(simd_data data, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_vector && info->nlanes > 0);

    PySIMDVectorObject *vec = PyObject_GC_New(PySIMDVectorObject, simd_active_state->vector_type);
    if (vec == NULL) {
        return (PySIMDVectorObject*)PyErr_NoMemory();
    }
    vec->dtype = dtype;
    if (info->is_bool) {
        // boolean vectors are internally treated as unsigned
        // vectors to add compatibility among all SIMD extensions
        switch(dtype) {
        case simd_data_vb8:
            data.vu8 = npyv_cvt_u8_b8(data.vb8);
            break;
        case simd_data_vb16:
            data.vu16 = npyv_cvt_u16_b16(data.vb16);
            break;
        case simd_data_vb32:
            data.vu32 = npyv_cvt_u32_b32(data.vb32);
            break;
        default:
            data.vu64 = npyv_cvt_u64_b64(data.vb64);
        }
    }
    npyv_store_u8(vec->data, data.vu8);
    PyObject_GC_Track(vec);
    return vec;
}

CYG_FINLINE simd_data
PySIMDVector_AsData(PySIMDVectorObject *vec, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_vector && info->nlanes > 0);

    simd_data data = {.u64 = 0};
    if (!PyObject_IsInstance(
        (PyObject *)vec, (PyObject *)simd_active_state->vector_type
    )) {
        PyErr_Format(PyExc_TypeError,
            "a vector type %s is required", info->pyname
        );
        return data;
    }
    if (vec->dtype != dtype) {
        PyErr_Format(PyExc_TypeError,
            "a vector type %s is required, got(%s)",
            info->pyname, simd_data_getinfo(vec->dtype)->pyname
        );
        return data;
    }

    data.vu8 = npyv_load_u8(vec->data);
    if (info->is_bool) {
        // boolean vectors are internally treated as unsigned
        // vectors to add compatibility among all SIMD extensions
        switch(dtype) {
        case simd_data_vb8:
            data.vb8 = npyv_cvt_b8_u8(data.vu8);
            break;
        case simd_data_vb16:
            data.vb16 = npyv_cvt_b16_u16(data.vu16);
            break;
        case simd_data_vb32:
            data.vb32 = npyv_cvt_b32_u32(data.vu32);
            break;
        default:
            data.vb64 = npyv_cvt_b64_u64(data.vu64);
        }
    }
    return data;
}

static int
PySIMDVectorType_Init(PyObject *module)
{
    simd_target_state *state = PyModule_GetState(module);
    state->vector_type = (PyTypeObject *)PyType_FromModuleAndSpec(
            module, &simd__vector_spec, NULL);
    if (state->vector_type == NULL) {
        return -1;
    }
    return PyModule_AddObjectRef(module, "vector_type", (PyObject *)state->vector_type);
}
