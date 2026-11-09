/*
 * This file is similar to the low-level loops for data type transfer
 * in `dtype_transfer.c` but for those which only require visiting
 * a single array (and mutating it in-place).
 *
 * As of writing, it is only used for CLEARing, which means mainly
 * Python object DECREF/dealloc followed by NULL'ing the data
 * (to support double clearing and ensure data is again in a usable state).
 * However, memory initialization and traverse follows similar
 * protocols (although traversal needs additional arguments).
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "alloc.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "dtype_traversal.h"


/* Buffer size with the same use case as the one in dtype_transfer.c */
#define NPY_LOWLEVEL_BUFFER_BLOCKSIZE  128


typedef int get_traverse_func_function(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *clear_info,
        NPY_ARRAYMETHOD_FLAGS *flags);

/*
 * Generic Clear function helpers:
 */

static int
get_clear_function(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *clear_info,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    NPY_traverse_info_init(clear_info);
    /* not that cleanup code bothers to check e.g. for floating point flags */
    *flags = PyArrayMethod_MINIMAL_FLAGS;

    PyArrayMethod_GetTraverseLoop *get_clear = NPY_DT_SLOTS(NPY_DTYPE(dtype))->get_clear_loop;
    if (get_clear == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                "Internal error, `get_clear_loop` not set for the DType '%S'",
                dtype);
        return -1;
    }

    if (get_clear(traverse_context, dtype, aligned, stride,
                  &clear_info->func, &clear_info->auxdata, flags) <  0) {
        /* callee should clean up, but make sure outside debug mode */
        assert(clear_info->func == NULL);
        clear_info->func = NULL;
        return -1;
    }
    Py_INCREF(dtype);
    clear_info->descr = dtype;

    return 0;
}

/*
 * Helper to set up a strided loop used for clearing.  Clearing means
 * deallocating any references (e.g. via Py_DECREF) and resetting the data
 * back into a usable/initialized state (e.g. by NULLing any references).
 *
 * The function will error when called on a dtype which does not have
 * references (and thus the get_clear_loop slot NULL).
 * Note that old-style user-dtypes use the "void" version.
 *
 * NOTE: This function may have a use for a `traverse_context` at some point
 *       but right now, it is always NULL and only exists to allow adding it
 *       in the future without changing the strided-loop signature.
 */
NPY_NO_EXPORT int
PyArray_GetClearFunction(
        int aligned, npy_intp stride, PyArray_Descr *dtype,
        NPY_traverse_info *clear_info, NPY_ARRAYMETHOD_FLAGS *flags)
{
    return get_clear_function(NULL, dtype, aligned, stride, clear_info, flags);
}


/*
 * Generic zerofill/fill function helper:
 */

static int
get_zerofill_function(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *zerofill_info,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    NPY_traverse_info_init(zerofill_info);
    /* not that filling code bothers to check e.g. for floating point flags */
    *flags = PyArrayMethod_MINIMAL_FLAGS;

    PyArrayMethod_GetTraverseLoop *get_zerofill = NPY_DT_SLOTS(NPY_DTYPE(dtype))->get_fill_zero_loop;
    if (get_zerofill == NULL) {
        /* Allowed to be NULL (and accept it here) */
        return 0;
    }

    if (get_zerofill(traverse_context, dtype, aligned, stride,
                     &zerofill_info->func, &zerofill_info->auxdata, flags) <  0) {
        /* callee should clean up, but make sure outside debug mode */
        assert(zerofill_info->func == NULL);
        zerofill_info->func = NULL;
        return -1;
    }
    if (zerofill_info->func == NULL) {
        /* Zerofill also may return func=NULL without an error. */
        return 0;
    }

    Py_INCREF(dtype);
    zerofill_info->descr = dtype;

    return 0;
}


/****************** Python Object clear ***********************/

static int
clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp size, npy_intp stride,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    PyObject *aligned_copy = NULL;
    while (size > 0) {
        /* Release the reference in src and set it to NULL */
        memcpy(&aligned_copy, data, sizeof(PyObject *));
        Py_XDECREF(aligned_copy);
        memset(data, 0, sizeof(PyObject *));

        data += stride;
        --size;
    }
    return 0;
}


NPY_NO_EXPORT int
npy_get_clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        int NPY_UNUSED(aligned), npy_intp NPY_UNUSED(fixed_stride),
        PyArrayMethod_TraverseLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI|NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &clear_object_strided_loop;
    return 0;
}


/**************** Python Object zero fill *********************/

static int
fill_zero_object_strided_loop(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp size, npy_intp stride,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    PyObject *zero = PyLong_FromLong(0);
    while (size--) {
        Py_INCREF(zero);
        // assumes `data` doesn't have a pre-existing object inside it
        memcpy(data, &zero, sizeof(zero));
        data += stride;
    }
    Py_DECREF(zero);
    return 0;
}

NPY_NO_EXPORT int
npy_object_get_fill_zero_loop(void *NPY_UNUSED(traverse_context),
                              const PyArray_Descr *NPY_UNUSED(descr),
                              int NPY_UNUSED(aligned),
                              npy_intp NPY_UNUSED(fixed_stride),
                              PyArrayMethod_TraverseLoop **out_loop,
                              NpyAuxData **NPY_UNUSED(out_auxdata),
                              NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &fill_zero_object_strided_loop;
    return 0;
}

/**************** Structured DType generic functionality ***************/

/*
 * Note that legacy user dtypes also make use of this.  Someone managed to
 * hack objects into them by adding a field that contains objects and this
 * remains (somewhat) valid.
 * (Unlike our voids, those fields must be hardcoded probably, but...)
 *
 * The below functionality mirrors the casting functionality relatively
 * closely.
 */

typedef struct {
    npy_intp src_offset;
    NPY_traverse_info info;
} single_field_traverse_data;

typedef struct {
    NpyAuxData base;
    npy_intp field_count;
    single_field_traverse_data fields[];
} fields_traverse_data;


/* traverse data free function */
static void
fields_traverse_data_free(NpyAuxData *data)
{
    fields_traverse_data *d = (fields_traverse_data *)data;

    for (npy_intp i = 0; i < d->field_count; ++i) {
        NPY_traverse_info_xfree(&d->fields[i].info);
    }
    PyMem_Free(d);
}


/* traverse data copy function (untested due to no direct use currently) */
static NpyAuxData *
fields_traverse_data_clone(NpyAuxData *data)
{
    fields_traverse_data *d = (fields_traverse_data *)data;

    npy_intp field_count = d->field_count;
    npy_intp structsize = sizeof(fields_traverse_data) +
                    field_count * sizeof(single_field_traverse_data);

    /* Allocate the data and populate it */
    fields_traverse_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;
    newdata->field_count = 0;

    /* Copy all the fields transfer data */
    single_field_traverse_data *in_field = d->fields;
    single_field_traverse_data *new_field = newdata->fields;

    for (; newdata->field_count < field_count;
                newdata->field_count++, in_field++, new_field++) {
        new_field->src_offset = in_field->src_offset;

        if (NPY_traverse_info_copy(&new_field->info, &in_field->info) < 0) {
            fields_traverse_data_free((NpyAuxData *)newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}


static int
traverse_fields_function(
        void *traverse_context, const _PyArray_LegacyDescr *NPY_UNUSED(descr),
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    fields_traverse_data *d = (fields_traverse_data *)auxdata;
    npy_intp i, field_count = d->field_count;

    /* Do the traversing a block at a time for better memory caching */
    const npy_intp blocksize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;

    for (;;) {
        if (N > blocksize) {
            for (i = 0; i < field_count; ++i) {
                single_field_traverse_data field = d->fields[i];
                if (field.info.func(traverse_context,
                        field.info.descr, data + field.src_offset,
                        blocksize, stride, field.info.auxdata) < 0) {
                    return -1;
                }
            }
            N -= blocksize;
            data += blocksize * stride;
        }
        else {
            for (i = 0; i < field_count; ++i) {
                single_field_traverse_data field = d->fields[i];
                if (field.info.func(traverse_context,
                        field.info.descr, data + field.src_offset,
                        N, stride, field.info.auxdata) < 0) {
                    return -1;
                }
            }
            return 0;
        }
    }
}


static int
get_fields_traverse_function(
        void *traverse_context, const _PyArray_LegacyDescr *dtype, int NPY_UNUSED(aligned),
        npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags,
        get_traverse_func_function *get_traverse_func)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *fld_dtype;
    npy_int i, structsize;
    Py_ssize_t field_count;

    names = dtype->names;
    field_count = PyTuple_GET_SIZE(names);

    /* Over-allocating here: less fields may be used */
    structsize = (sizeof(fields_traverse_data) +
                    field_count * sizeof(single_field_traverse_data));
    /* Allocate the data and populate it */
    fields_traverse_data *data = PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    data->base.free = &fields_traverse_data_free;
    data->base.clone = &fields_traverse_data_clone;
    data->field_count = 0;

    single_field_traverse_data *field = data->fields;
    for (i = 0; i < field_count; ++i) {
        int offset;

        key = PyTuple_GET_ITEM(names, i);
        tup = PyDict_GetItem(dtype->fields, key); // noqa: borrowed-ref OK
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return -1;
        }
        if (get_traverse_func == &get_clear_function
                && !PyDataType_REFCHK(fld_dtype)) {
            /* No need to do clearing (could change to use NULL return) */
            continue;
        }
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        if (get_traverse_func(
                traverse_context, fld_dtype, 0,
                stride, &field->info, &clear_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return -1;
        }
        if (field->info.func == NULL) {
            /* zerofill allows NULL func as "default" memset to zero */
            continue;
        }
        *flags = PyArrayMethod_COMBINED_FLAGS(*flags, clear_flags);
        field->src_offset = offset;
        data->field_count++;
        field++;
    }

    *out_func = (PyArrayMethod_TraverseLoop *)&traverse_fields_function;
    *out_auxdata = (NpyAuxData *)data;

    return 0;
}


typedef struct {
    NpyAuxData base;
    npy_intp count;
    NPY_traverse_info info;
} subarray_traverse_data;


/* traverse data free function */
static void
subarray_traverse_data_free(NpyAuxData *data)
{
    subarray_traverse_data *d = (subarray_traverse_data *)data;

    NPY_traverse_info_xfree(&d->info);
    PyMem_Free(d);
}


/* traverse data copy function */
static NpyAuxData *
subarray_traverse_data_clone(NpyAuxData *data)
{
    subarray_traverse_data *d = (subarray_traverse_data *)data;

    /* Allocate the data and populate it */
    subarray_traverse_data *newdata = PyMem_Malloc(sizeof(subarray_traverse_data));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;
    newdata->count = d->count;

    if (NPY_traverse_info_copy(&newdata->info, &d->info) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }

    return (NpyAuxData *)newdata;
}


static int
traverse_subarray_func(
        void *traverse_context, const PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    subarray_traverse_data *subarr_data = (subarray_traverse_data *)auxdata;

    PyArrayMethod_TraverseLoop *func = subarr_data->info.func;
    const PyArray_Descr *sub_descr = subarr_data->info.descr;
    npy_intp sub_N = subarr_data->count;
    NpyAuxData *sub_auxdata = subarr_data->info.auxdata;
    npy_intp sub_stride = sub_descr->elsize;

    while (N--) {
        if (func(traverse_context, sub_descr, data,
                 sub_N, sub_stride, sub_auxdata) < 0) {
            return -1;
        }
        data += stride;
    }
    return 0;
}


static int
get_subarray_traverse_func(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp size, npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags,
        get_traverse_func_function *get_traverse_func)
{
    subarray_traverse_data *auxdata = PyMem_Malloc(sizeof(subarray_traverse_data));
    if (auxdata == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    auxdata->count = size;
    auxdata->base.free = &subarray_traverse_data_free;
    auxdata->base.clone = &subarray_traverse_data_clone;

    if (get_traverse_func(
            traverse_context, dtype, aligned,
            dtype->elsize, &auxdata->info, flags) < 0) {
        PyMem_Free(auxdata);
        return -1;
    }
    if (auxdata->info.func == NULL) {
        /* zerofill allows func to be NULL, in which we need not do anything */
        PyMem_Free(auxdata);
        *out_func = NULL;
        *out_auxdata = NULL;
        return 0;
    }
    *out_func = &traverse_subarray_func;
    *out_auxdata = (NpyAuxData *)auxdata;

    return 0;
}


static int
clear_no_op(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        char *NPY_UNUSED(data), npy_intp NPY_UNUSED(size),
        npy_intp NPY_UNUSED(stride), NpyAuxData *NPY_UNUSED(auxdata))
{
    return 0;
}

NPY_NO_EXPORT int
npy_get_clear_void_and_legacy_user_dtype_loop(
        void *traverse_context, const _PyArray_LegacyDescr *dtype, int aligned,
        npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    /*
     * If there are no references, it's a nop.  This path should not be hit
     * but structured dtypes are tricky when a dtype which included references
     * was sliced to not include any.
     */
    if (!PyDataType_REFCHK((PyArray_Descr *)dtype)) {
        *out_func = &clear_no_op;
        return 0;
    }

    if (dtype->subarray != NULL) {
        PyArray_Dims shape = {NULL, -1};
        npy_intp size;

        if (!(PyArray_IntpConverter(dtype->subarray->shape, &shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return -1;
        }
        size = PyArray_MultiplyList(shape.ptr, shape.len);
        npy_free_cache_dim_obj(shape);

        if (get_subarray_traverse_func(
                traverse_context, dtype->subarray->base, aligned, size, stride,
                out_func, out_auxdata, flags, &get_clear_function) < 0) {
            return -1;
        }

        return 0;
    }
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dtype)) {
        if (get_fields_traverse_function(
                traverse_context, dtype, aligned, stride,
                out_func, out_auxdata, flags, &get_clear_function) < 0) {
            return -1;
        }
        return 0;
    }
    else if (dtype->type_num == NPY_VOID) {
        /* 
         * Void dtypes can have "ghosts" of objects marking the dtype because
         * holes (or the raw bytes if fields are gone) may include objects.
         * Paths that need those flags should probably be considered incorrect.
         * But as long as this can happen (a V8 that indicates references)
         * we need to make it a no-op here.
         */
        *out_func = &clear_no_op;
        return 0;
    }

    PyErr_Format(PyExc_RuntimeError,
            "Internal error, tried to fetch clear function for the "
            "user dtype '%S' without fields or subarray (legacy support).",
            dtype);
    return -1;
}

/**************** Structured DType zero fill ***************/


static int
zerofill_fields_function(
        void *traverse_context, const _PyArray_LegacyDescr *descr,
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    npy_intp itemsize = descr->elsize;

    /*
     * TODO: We could optimize this by chunking, but since we currently memset
     *       each element always, just loop manually.
     */
    while (N--) {
        memset(data, 0, itemsize);
        if (traverse_fields_function(
                traverse_context, descr, data, 1, stride, auxdata) < 0) {
            return -1;
        }
        data +=stride;
    }
    return 0;
}

/*
 * Similar to other (e.g. clear) traversal loop getter, but unlike it, we
 * do need to take care of zeroing out everything (in principle not gaps).
 * So we add a memset before calling the actual traverse function for the
 * structured path.
 */
NPY_NO_EXPORT int
npy_get_zerofill_void_and_legacy_user_dtype_loop(
        void *traverse_context, const _PyArray_LegacyDescr *dtype, int aligned,
        npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (dtype->subarray != NULL) {
        PyArray_Dims shape = {NULL, -1};
        npy_intp size;

        if (!(PyArray_IntpConverter(dtype->subarray->shape, &shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return -1;
        }
        size = PyArray_MultiplyList(shape.ptr, shape.len);
        npy_free_cache_dim_obj(shape);

        if (get_subarray_traverse_func(
                traverse_context, dtype->subarray->base, aligned, size, stride,
                out_func, out_auxdata, flags, &get_zerofill_function) < 0) {
            return -1;
        }

        return 0;
    }
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dtype)) {
        if (get_fields_traverse_function(
                traverse_context, dtype, aligned, stride,
                out_func, out_auxdata, flags, &get_zerofill_function) < 0) {
            return -1;
        }
        if (((fields_traverse_data *)*out_auxdata)->field_count == 0) {
            /* If there are no fields, just return NULL for zerofill */
            NPY_AUXDATA_FREE(*out_auxdata);
            *out_auxdata = NULL;
            *out_func = NULL;
            return 0;
        }
        /* 
         * Traversal skips fields that have no custom zeroing, so we need
         * to take care of it.
         */
        *out_func = (PyArrayMethod_TraverseLoop *)&zerofill_fields_function;
        return 0;
    }

    /* Otherwise, assume there is nothing to do (user dtypes reach here) */
    *out_auxdata = NULL;
    *out_func = NULL;
    return 0;
}


/* Immutable native reference layout. No node retains a Python object. */
enum npy_object_ref_kind {
    NPY_OBJECT_REF_NONE,
    NPY_OBJECT_REF_OBJECT,
    NPY_OBJECT_REF_GROUP,
    NPY_OBJECT_REF_REPEAT,
};

typedef struct {
    enum npy_object_ref_kind kind;
    npy_intp offset;
    npy_intp nrefs;
    npy_intp first_child;
    npy_intp next_sibling;
    npy_intp count;
    npy_intp stride;
} npy_object_ref_node;

typedef struct {
    npy_intp child;
    npy_intp upper;
} npy_object_ref_edge;

struct npy_object_reference_plan {
    npy_object_ref_node *nodes;
    npy_object_ref_edge *edges;
    npy_intp length;
    npy_intp capacity;
};

typedef struct {
    PyArray_Descr *descr;  /* Borrowed only while the builder runs. */
    npy_intp node;
    npy_intp index;
    npy_intp count;
    npy_intp last_child;
    int initialized;
} npy_object_ref_build_frame;

NPY_NO_EXPORT void
npy_object_reference_plan_free(npy_object_reference_plan *plan)
{
    if (plan != NULL) {
        PyMem_RawFree(plan->nodes);
        PyMem_RawFree(plan->edges);
        PyMem_RawFree(plan);
    }
}

static int
object_ref_plan_append(npy_object_reference_plan *plan, npy_intp offset,
                       npy_intp *index)
{
    if (plan->length == plan->capacity) {
        if (plan->capacity > NPY_MAX_INTP / (2 * (npy_intp)sizeof(*plan->nodes))) {
            PyErr_NoMemory();
            return -1;
        }
        npy_intp capacity = plan->capacity == 0 ? 16 : 2 * plan->capacity;
        npy_object_ref_node *nodes = PyMem_RawRealloc(
                plan->nodes, capacity * sizeof(*nodes));
        if (nodes == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        plan->nodes = nodes;
        plan->capacity = capacity;
    }
    *index = plan->length++;
    plan->nodes[*index] = (npy_object_ref_node){
        .offset = offset, .first_child = -1, .next_sibling = -1,
    };
    return 0;
}

NPY_NO_EXPORT int
npy_object_reference_plan_new(PyArray_Descr *descr,
                              npy_object_reference_plan **result)
{
    *result = NULL;
    if (!PyDataType_ISLEGACY(descr) || !PyDataType_REFCHK(descr)) {
        return 0;
    }
    npy_object_reference_plan *plan = PyMem_RawCalloc(1, sizeof(*plan));
    if (plan == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    npy_object_ref_build_frame local[16];
    npy_object_ref_build_frame *frames = local;
    npy_intp capacity = Py_ARRAY_LENGTH(local), depth = 1;
    npy_intp root;
    if (object_ref_plan_append(plan, 0, &root) < 0) {
        goto fail;
    }
    frames[0] = (npy_object_ref_build_frame){
        .descr = descr, .node = root, .last_child = -1,
    };
    while (depth > 0) {
        npy_object_ref_build_frame *frame = &frames[depth - 1];
        npy_object_ref_node *node = &plan->nodes[frame->node];
        if (!frame->initialized) {
            frame->initialized = 1;
            if (frame->descr->type_num == NPY_OBJECT) {
                if (frame->descr->elsize < (npy_intp)sizeof(PyObject *)) {
                    PyErr_SetString(PyExc_ValueError, "invalid object reference size");
                    goto fail;
                }
                node->kind = NPY_OBJECT_REF_OBJECT;
                node->nrefs = 1;
            }
            else if (PyDataType_ISLEGACY(frame->descr)
                    && PyDataType_REFCHK(frame->descr)) {
                _PyArray_LegacyDescr *legacy = (_PyArray_LegacyDescr *)frame->descr;
                if (legacy->subarray != NULL && legacy->subarray->base->elsize > 0) {
                    npy_intp step = legacy->subarray->base->elsize;
                    if (legacy->elsize < 0 || legacy->elsize % step != 0) {
                        PyErr_SetString(PyExc_ValueError, "invalid object subarray layout");
                        goto fail;
                    }
                    node->kind = NPY_OBJECT_REF_REPEAT;
                    node->stride = step;
                    node->count = legacy->elsize / step;
                    frame->count = node->count == 0 ? 0 : 1;
                }
                else if (legacy->names != NULL) {
                    node->kind = NPY_OBJECT_REF_GROUP;
                    frame->count = PyTuple_GET_SIZE(legacy->names);
                }
            }
        }
        if (frame->index == frame->count) {
            if (node->kind == NPY_OBJECT_REF_REPEAT && node->nrefs != 0) {
                if (node->count > NPY_MAX_INTP / node->nrefs) {
                    PyErr_SetString(PyExc_OverflowError, "object reference count overflow");
                    goto fail;
                }
                node->nrefs *= node->count;
            }
            npy_intp count = node->nrefs;
            depth--;
            if (depth != 0) {
                npy_object_ref_node *parent = &plan->nodes[frames[depth - 1].node];
                if (parent->nrefs > NPY_MAX_INTP - count) {
                    PyErr_SetString(PyExc_OverflowError, "object reference count overflow");
                    goto fail;
                }
                parent->nrefs += count;
            }
            continue;
        }
        _PyArray_LegacyDescr *legacy = (_PyArray_LegacyDescr *)frame->descr;
        PyArray_Descr *child_descr;
        npy_intp offset = 0;
        if (node->kind == NPY_OBJECT_REF_REPEAT) {
            child_descr = legacy->subarray->base;
        }
        else {
            PyObject *name = PyTuple_GET_ITEM(legacy->names, frame->index);
            PyObject *field = legacy->fields == NULL ? NULL :
                    PyDict_GetItemWithError(legacy->fields, name);
            if (field == NULL || !PyTuple_Check(field) || PyTuple_GET_SIZE(field) < 2) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError, "incomplete structured reference layout");
                }
                goto fail;
            }
            child_descr = (PyArray_Descr *)PyTuple_GET_ITEM(field, 0);
            offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(field, 1));
            if (offset == -1 && PyErr_Occurred()) {
                goto fail;
            }
            if (offset < 0 || child_descr->elsize < 0
                    || offset > frame->descr->elsize
                    || child_descr->elsize > frame->descr->elsize - offset) {
                PyErr_SetString(PyExc_ValueError, "object field outside descriptor layout");
                goto fail;
            }
        }
        frame->index++;
        if (!PyDataType_ISLEGACY(child_descr) || !PyDataType_REFCHK(child_descr)) {
            continue;
        }
        for (npy_intp i = 0; i < depth; i++) {
            if (frames[i].descr == child_descr) {
                PyErr_SetString(PyExc_ValueError, "cyclic descriptor reference layout");
                goto fail;
            }
        }
        npy_intp child;
        if (object_ref_plan_append(plan, offset, &child) < 0) {
            goto fail;
        }
        /* Appending may move the node buffer. */
        node = &plan->nodes[frame->node];
        if (frame->last_child == -1) {
            node->first_child = child;
        }
        else {
            plan->nodes[frame->last_child].next_sibling = child;
        }
        frame->last_child = child;
        if (depth == capacity) {
            if (capacity > NPY_MAX_INTP / (2 * (npy_intp)sizeof(*frames))) {
                PyErr_NoMemory();
                goto fail;
            }
            npy_intp larger_capacity = 2 * capacity;
            npy_object_ref_build_frame *larger = PyMem_RawMalloc(
                    larger_capacity * sizeof(*larger));
            if (larger == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
            memcpy(larger, frames, depth * sizeof(*frames));
            if (frames != local) {
                PyMem_RawFree(frames);
            }
            frames = larger;
            capacity = larger_capacity;
        }
        frames[depth++] = (npy_object_ref_build_frame){
            .descr = child_descr, .node = child, .last_child = -1,
        };
    }
    if (frames != local) {
        PyMem_RawFree(frames);
    }
    if (plan->nodes[0].nrefs == 0) {
        /* A valid empty legacy layout must not fall back to Python fields. */
        *result = plan;
    }
    else {
        if (plan->length > NPY_MAX_INTP / (npy_intp)sizeof(*plan->edges)) {
            PyErr_NoMemory();
            npy_object_reference_plan_free(plan);
            return -1;
        }
        plan->edges = PyMem_RawMalloc(plan->length * sizeof(*plan->edges));
        if (plan->edges == NULL) {
            PyErr_NoMemory();
            npy_object_reference_plan_free(plan);
            return -1;
        }
        npy_intp edge_count = 0;
        for (npy_intp i = 0; i < plan->length; i++) {
            npy_object_ref_node *node = &plan->nodes[i];
            if (node->kind != NPY_OBJECT_REF_GROUP) {
                continue;
            }
            /* Dense ordinal indexes avoid a quadratic scan for wide records. */
            node->stride = edge_count;
            node->count = 0;
            npy_intp upper = 0;
            for (npy_intp child = node->first_child; child != -1;
                    child = plan->nodes[child].next_sibling) {
                upper += plan->nodes[child].nrefs;
                plan->edges[edge_count++] = (npy_object_ref_edge){child, upper};
                node->count++;
            }
        }
        *result = plan;
    }
    return 0;
fail:
    if (frames != local) {
        PyMem_RawFree(frames);
    }
    npy_object_reference_plan_free(plan);
    return -1;
}

/* Descend using subtree counts instead of a mutable execution stack. */
static npy_intp
object_ref_plan_offset(const npy_object_reference_plan *plan, npy_intp ordinal)
{
    const npy_object_ref_node *node = &plan->nodes[0];
    npy_intp offset = 0;
    for (;;) {
        offset += node->offset;
        if (node->kind == NPY_OBJECT_REF_OBJECT) {
            return offset;
        }
        const npy_object_ref_node *child = &plan->nodes[node->first_child];
        if (node->kind == NPY_OBJECT_REF_REPEAT) {
            offset += (ordinal / child->nrefs) * node->stride;
            ordinal %= child->nrefs;
        }
        else {
            const npy_object_ref_edge *edges = plan->edges + node->stride;
            npy_intp low = 0, high = node->count;
            while (low < high) {
                npy_intp middle = low + (high - low) / 2;
                if (ordinal >= edges[middle].upper) {
                    low = middle + 1;
                }
                else {
                    high = middle;
                }
            }
            if (low > 0) {
                ordinal -= edges[low - 1].upper;
            }
            child = &plan->nodes[edges[low].child];
        }
        node = child;
    }
}

NPY_NO_EXPORT int
npy_object_reference_plan_apply_item(const npy_object_reference_plan *plan,
                                     char *data, visitproc visit, void *arg)
{
    npy_intp nrefs = plan->nodes[0].nrefs;
    for (npy_intp j = 0; j < nrefs; j++) {
        char *address = data + object_ref_plan_offset(plan, j);
        PyObject *object;
        memcpy(&object, address, sizeof(object));
        if (visit == NULL) {
            /* A destructor may reenter: relinquish ownership first. */
            memset(address, 0, sizeof(object));
            Py_XDECREF(object);
        }
        else if (object != NULL) {
            int result = visit(object, arg);
            if (result != 0) {
                return result;
            }
        }
    }
    return 0;
}

NPY_NO_EXPORT int
npy_object_reference_plan_apply(const npy_object_reference_plan *plan,
                                PyArrayObject *array, visitproc visit, void *arg)
{
    npy_intp nrefs = plan->nodes[0].nrefs;
    if (nrefs == 0) {
        return 0;
    }
    npy_intp coordinates[NPY_MAXDIMS] = {0};
    char *data = PyArray_BYTES(array);
    npy_intp size = PyArray_SIZE(array);
    for (npy_intp i = 0; i < size; i++) {
        int result = npy_object_reference_plan_apply_item(plan, data, visit, arg);
        if (result != 0) {
            return result;
        }
        for (int axis = PyArray_NDIM(array) - 1; axis >= 0; axis--) {
            coordinates[axis]++;
            data += PyArray_STRIDES(array)[axis];
            if (coordinates[axis] < PyArray_DIMS(array)[axis]) {
                break;
            }
            data -= coordinates[axis] * PyArray_STRIDES(array)[axis];
            coordinates[axis] = 0;
        }
    }
    return 0;
}
