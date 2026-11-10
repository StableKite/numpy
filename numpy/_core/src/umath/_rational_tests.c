/* Fixed size rational numbers exposed to Python */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Build as a downstream module targeting NumPy 2.0 to exercise the
 * public DType API and the `NPY_DT_legacy_descriptor_proto` backport. */
#if defined(NPY_INTERNAL_BUILD)
#undef NPY_INTERNAL_BUILD
#endif
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION

#include "numpy/interpreter_context.h"
typedef struct {
    NPY_SI_Context numpy;
    int runtime_version;
    PyTypeObject *rational;
    PyTypeObject *rational2;
    PyTypeObject *dtype2;
    PyObject *shared_owner;
    int initialized;
} rational_module_state;
static _Thread_local rational_module_state *rational_active;
#define NPY_API_MODULE_CONTEXT (&rational_active->numpy)
#define NPY_API_MODULE_RUNTIME_VERSION (rational_active->runtime_version)
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
#include "numpy/npy_3kcompat.h"
#include "_owned_legacy_test_loop.h"

static struct PyModuleDef moduledef;

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

#include <math.h>


/* Relevant arithmetic exceptions */

/* Legacy integer gcd/lcm loops may be detached. Their unchanged legacy
 * selector captures the actual caller before detach, as for _umath_tests. */
static void
set_arithmetic_error(PyObject *exception, const char *message)
{
#if PY_VERSION_HEX >= 0x030d0000
    PyThreadState *attached = PyThreadState_GetUnchecked();
    if (attached == NULL) {
        assert(owned_legacy_test_caller != NULL);
        PyEval_RestoreThread(owned_legacy_test_caller);
    }
    if (!PyErr_Occurred()) {
        PyErr_SetString(exception, message);
    }
    if (attached == NULL) {
        PyThreadState *caller = PyEval_SaveThread();
        assert(caller == owned_legacy_test_caller);
    }
#else
    /* Before 3.13 this module retains its main-interpreter-only contract.
     * Numeric gcd/lcm loops may still release that interpreter's GIL. */
    PyGILState_STATE saved = PyGILState_Ensure();
    if (!PyErr_Occurred()) {
        PyErr_SetString(exception, message);
    }
    PyGILState_Release(saved);
#endif
}
static void
set_overflow(void)
{
    set_arithmetic_error(PyExc_OverflowError, "overflow in rational arithmetic");
}
static void
set_zero_divide(void)
{
    set_arithmetic_error(PyExc_ZeroDivisionError, "zero divide in rational arithmetic");
}

/* Integer arithmetic utilities */

static inline npy_int32
safe_neg(npy_int32 x)
{
    if (x == NPY_MIN_INT32) {
        set_overflow();
        return 0;
    }
    return -x;
}
static inline npy_int32
safe_abs32(npy_int32 x)
{
    if (x == NPY_MIN_INT32) {
        set_overflow();
        return 0;
    }
    return x < 0 ? -x : x;
}
static inline npy_int64
safe_abs64(npy_int64 x)
{
    if (x == NPY_MIN_INT64) {
        set_overflow();
        return 0;
    }
    return x < 0 ? -x : x;
}

static inline npy_int64
gcd(npy_int64 x, npy_int64 y) {
    x = safe_abs64(x);
    y = safe_abs64(y);
    if (x < y) {
        npy_int64 t = x;
        x = y;
        y = t;
    }
    while (y) {
        npy_int64 t;
        x = x%y;
        t = x;
        x = y;
        y = t;
    }
    return x;
}

static inline npy_int64
lcm(npy_int64 x, npy_int64 y)
{
    if (!x || !y) {
        return 0;
    }
    x = safe_abs64(x);
    y = safe_abs64(y);
    if (!x || !y) {
        return 0;  /* an absolute-value overflow already set the exception */
    }
    x /= gcd(x, y);
    if (x > NPY_MAX_INT64 / y) {
        set_overflow();
        return 0;
    }
    return x * y;
}

/* Fixed precision rational numbers */

typedef struct {
    /* numerator */
    npy_int32 n;
    /*
     * denominator minus one: numpy.zeros() uses memset(0) for non-object
     * types, so need to ensure that rational(0) has all zero bytes
     */
    npy_int32 dmm;
} rational;

static inline rational
make_rational_int(npy_int64 n) {
    rational r = {(npy_int32)n,0};
    if (r.n != n) {
        set_overflow();
    }
    return r;
}

static rational
make_rational_slow(npy_int64 n_, npy_int64 d_) {
    rational r = {0};
    if (!d_) {
        set_zero_divide();
    }
    else {
        npy_int64 g = gcd(n_,d_);
        npy_int32 d;
        n_ /= g;
        d_ /= g;
        r.n = (npy_int32)n_;
        d = (npy_int32)d_;
        if (r.n!=n_ || d!=d_) {
            set_overflow();
        }
        else {
            if (d <= 0) {
                d = -d;
                r.n = safe_neg(r.n);
            }
            r.dmm = d-1;
        }
    }
    return r;
}

static inline npy_int32
d(rational r) {
    return r.dmm+1;
}

/* Assumes d_ > 0 */
static rational
make_rational_fast(npy_int64 n_, npy_int64 d_) {
    npy_int64 g = gcd(n_,d_);
    rational r;
    n_ /= g;
    d_ /= g;
    r.n = (npy_int32)n_;
    r.dmm = (npy_int32)(d_-1);
    if (r.n!=n_ || r.dmm+1!=d_) {
        set_overflow();
    }
    return r;
}

static inline rational
rational_negative(rational r) {
    rational x;
    x.n = safe_neg(r.n);
    x.dmm = r.dmm;
    return x;
}

static inline rational
rational_add(rational x, rational y) {
    /*
     * Note that the numerator computation can never overflow int128_t,
     * since each term is strictly under 2**128/4 (since d > 0).
     */
    return make_rational_fast((npy_int64)x.n*d(y)+(npy_int64)d(x)*y.n,
        (npy_int64)d(x)*d(y));
}

static inline rational
rational_subtract(rational x, rational y) {
    /* We're safe from overflow as with + */
    return make_rational_fast((npy_int64)x.n*d(y)-(npy_int64)d(x)*y.n,
        (npy_int64)d(x)*d(y));
}

static inline rational
rational_multiply(rational x, rational y) {
    /* We're safe from overflow as with + */
    return make_rational_fast((npy_int64)x.n*y.n,(npy_int64)d(x)*d(y));
}

static inline rational
rational_divide(rational x, rational y) {
    return make_rational_slow((npy_int64)x.n*d(y),(npy_int64)d(x)*y.n);
}

static inline npy_int64
rational_floor(rational x) {
    /* Always round down */
    if (x.n>=0) {
        return x.n/d(x);
    }
    /*
     * This can be done without casting up to 64 bits, but it requires
     * working out all the sign cases
     */
    return -((-(npy_int64)x.n+d(x)-1)/d(x));
}

static inline npy_int64
rational_ceil(rational x) {
    return -rational_floor(rational_negative(x));
}

static inline rational
rational_remainder(rational x, rational y) {
    return rational_subtract(x, rational_multiply(y,make_rational_int(
                    rational_floor(rational_divide(x,y)))));
}

static inline rational
rational_abs(rational x) {
    rational y;
    y.n = safe_abs32(x.n);
    y.dmm = x.dmm;
    return y;
}

static inline npy_int64
rational_rint(rational x) {
    /*
     * Round towards nearest integer, moving exact half integers towards
     * zero
     */
    npy_int32 d_ = d(x);
    return (2*(npy_int64)x.n+(x.n<0?-d_:d_))/(2*(npy_int64)d_);
}

static inline int
rational_sign(rational x) {
    return x.n<0?-1:x.n==0?0:1;
}

static inline rational
rational_inverse(rational x) {
    rational y = {0};
    if (!x.n) {
        set_zero_divide();
    }
    else {
        npy_int32 d_;
        y.n = d(x);
        d_ = x.n;
        if (d_ <= 0) {
            d_ = safe_neg(d_);
            y.n = -y.n;
        }
        y.dmm = d_-1;
    }
    return y;
}

static inline int
rational_eq(rational x, rational y) {
    /*
     * Since we enforce d > 0, and store fractions in reduced form,
     * equality is easy.
     */
    return x.n==y.n && x.dmm==y.dmm;
}

static inline int
rational_ne(rational x, rational y) {
    return !rational_eq(x,y);
}

static inline int
rational_lt(rational x, rational y) {
    return (npy_int64)x.n*d(y) < (npy_int64)y.n*d(x);
}

static inline int
rational_gt(rational x, rational y) {
    return rational_lt(y,x);
}

static inline int
rational_le(rational x, rational y) {
    return !rational_lt(y,x);
}

static inline int
rational_ge(rational x, rational y) {
    return !rational_lt(x,y);
}

static inline npy_int32
rational_int(rational x) {
    return x.n/d(x);
}

static inline double
rational_double(rational x) {
    return (double)x.n/d(x);
}

static inline int
rational_nonzero(rational x) {
    return x.n!=0;
}

static int
scan_rational(const char** s, rational* x) {
    long n,d;
    int offset;
    const char* ss;
    if (sscanf(*s,"%ld%n",&n,&offset)<=0) {
        return 0;
    }
    ss = *s+offset;
    if (*ss!='/') {
        *s = ss;
        *x = make_rational_int(n);
        return 1;
    }
    ss++;
    if (sscanf(ss,"%ld%n",&d,&offset)<=0 || d<=0) {
        return 0;
    }
    *s = ss+offset;
    *x = make_rational_slow(n,d);
    return 1;
}

/* Expose rational to Python as a numpy scalar */

typedef struct {
    PyObject_HEAD
    rational r;
} PyRational;

static PyObject *pyrational_new(PyTypeObject *, PyObject *, PyObject *);
#define PyRational_Type (*rational_active->rational)
#define PyRational2_Type (*rational_active->rational2)
#define NPY_Rational2DType (*(PyArray_DTypeMeta *)rational_active->dtype2)

static inline int
PyRational_Check(PyObject *object)
{
    /* A real C base, including Python subclasses; no foreign cached type. */
    for (PyTypeObject *type = Py_TYPE(object); type != NULL; type = type->tp_base) {
        if (type->tp_new == pyrational_new) {
            return 1;
        }
    }
    return 0;
}
static rational_module_state *
rational_state_from_scalar(PyObject *object)
{
    for (PyTypeObject *type = Py_TYPE(object); type != NULL; type = type->tp_base) {
        if (type->tp_new == pyrational_new) {
            rational_module_state *state = PyType_GetModuleState(type);
            if (state != NULL) {
                return state;
            }
            PyErr_Clear();  /* an ordinary Python subclass has no C module */
        }
    }
    PyErr_SetString(PyExc_TypeError, "rational scalar has no owning module");
    return NULL;
}
static PyTypeObject *
rational_result_type(PyObject *left, PyObject *right)
{
    PyObject *scalar = PyRational_Check(left) ? left : right;
    rational_module_state *state = rational_state_from_scalar(scalar);
    return state == NULL ? NULL : state->rational;
}

/* Registration is canonical within one interpreter, like the registered
 * dtype itself. Reimports reuse its actual types/loops. The interpreter owns
 * this cache and releases it at close; there is no process-wide PyObject. */
static const char *rational_cache_key = "numpy._core._rational_tests.owned_v1";
static PyObject *
rational_cached_module(void)
{
    PyObject *dict = PyInterpreterState_GetDict(PyThreadState_GetInterpreter(PyThreadState_Get()));
    if (dict == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "rational interpreter dictionary unavailable");
        return NULL;
    }
    return PyDict_GetItemString(dict, rational_cache_key);  /* borrowed */
}
static rational_module_state *
rational_registered_state(void)
{
    PyObject *module = rational_cached_module();
    if (module == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "rational context is not initialized");
        }
        return NULL;
    }
    if (!PyModule_Check(module) || PyModule_GetDef(module) != &moduledef) {
        PyErr_SetString(PyExc_RuntimeError, "invalid rational interpreter owner");
        return NULL;
    }
    return PyModule_GetState(module);
}

static PyObject*
PyRational_FromRational(PyTypeObject* type, rational x) {
    PyRational* p = (PyRational*)type->tp_alloc(type, 0);
    if (p) {
        p->r = x;
    }
    return (PyObject*)p;
}

static PyObject*
pyrational_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    Py_ssize_t size;
    PyObject* x[2];
    long n[2]={0,1};
    int i;
    rational r;
    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                "constructor takes no keyword arguments");
        return 0;
    }
    size = PyTuple_GET_SIZE(args);
    if (size > 2) {
        PyErr_SetString(PyExc_TypeError,
                "expected rational or numerator and optional denominator");
        return 0;
    }

    if (size == 1) {
        x[0] = PyTuple_GET_ITEM(args, 0);
        if (PyRational_Check(x[0])) {
            if (Py_TYPE(x[0]) == type) {
                Py_INCREF(x[0]);
                return x[0];
            }
            return PyRational_FromRational(type, ((PyRational*)x[0])->r);
        }
        // TODO: allow construction from unicode strings
        else if (PyBytes_Check(x[0])) {
            const char* s = PyBytes_AS_STRING(x[0]);
            if (scan_rational(&s,&r)) {
                const char* p;
                for (p = s; *p; p++) {
                    if (!isspace(*p)) {
                        goto bad;
                    }
                }
                return PyRational_FromRational(type, r);
            }
            bad:
            PyErr_Format(PyExc_ValueError,
                    "invalid rational literal '%s'",s);
            return 0;
        }
    }

    for (i=0; i<size; i++) {
        PyObject* y;
        int eq;
        x[i] = PyTuple_GET_ITEM(args, i);
        n[i] = PyLong_AsLong(x[i]);
        if (error_converting(n[i])) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                        "expected integer %s, got %s",
                        (i ? "denominator" : "numerator"),
                        x[i]->ob_type->tp_name);
            }
            return 0;
        }
        /* Check that we had an exact integer */
        y = PyLong_FromLong(n[i]);
        if (!y) {
            return 0;
        }
        eq = PyObject_RichCompareBool(x[i],y,Py_EQ);
        Py_DECREF(y);
        if (eq<0) {
            return 0;
        }
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected integer %s, got %s",
                    (i ? "denominator" : "numerator"),
                    x[i]->ob_type->tp_name);
            return 0;
        }
    }
    r = make_rational_slow(n[0],n[1]);
    if (PyErr_Occurred()) {
        return 0;
    }
    return PyRational_FromRational(type, r);
}

/*
 * Returns Py_NotImplemented on most conversion failures, or raises an
 * overflow error for too long ints
 */
#define AS_RATIONAL(dst,object) \
    { \
        dst.n = 0; \
        if (PyRational_Check(object)) { \
            dst = ((PyRational*)object)->r; \
        } \
        else { \
            PyObject* y_; \
            int eq_; \
            long n_ = PyLong_AsLong(object); \
            if (error_converting(n_)) { \
                if (PyErr_ExceptionMatches(PyExc_TypeError)) { \
                    PyErr_Clear(); \
                    Py_INCREF(Py_NotImplemented); \
                    return Py_NotImplemented; \
                } \
                return 0; \
            } \
            y_ = PyLong_FromLong(n_); \
            if (!y_) { \
                return 0; \
            } \
            eq_ = PyObject_RichCompareBool(object,y_,Py_EQ); \
            Py_DECREF(y_); \
            if (eq_<0) { \
                return 0; \
            } \
            if (!eq_) { \
                Py_INCREF(Py_NotImplemented); \
                return Py_NotImplemented; \
            } \
            dst = make_rational_int(n_); \
        } \
    }

static PyObject*
pyrational_richcompare(PyObject* a, PyObject* b, int op) {
    rational x, y;
    int result = 0;
    AS_RATIONAL(x,a);
    AS_RATIONAL(y,b);
    #define OP(py,op) case py: result = rational_##op(x,y); break;
    switch (op) {
        OP(Py_LT,lt)
        OP(Py_LE,le)
        OP(Py_EQ,eq)
        OP(Py_NE,ne)
        OP(Py_GT,gt)
        OP(Py_GE,ge)
    };
    #undef OP
    return PyBool_FromLong(result);
}

static PyObject*
pyrational_repr(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    if (d(x)!=1) {
        return PyUnicode_FromFormat(
                "rational(%ld,%ld)",(long)x.n,(long)d(x));
    }
    else {
        return PyUnicode_FromFormat(
                "rational(%ld)",(long)x.n);
    }
}

static PyObject*
pyrational_str(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    if (d(x)!=1) {
        return PyUnicode_FromFormat(
                "%ld/%ld",(long)x.n,(long)d(x));
    }
    else {
        return PyUnicode_FromFormat(
                "%ld",(long)x.n);
    }
}

static npy_hash_t
pyrational_hash(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    /* Use a fairly weak hash as Python expects */
    long h = 131071*x.n+524287*x.dmm;
    /* Never return the special error value -1 */
    return h==-1?2:h;
}

#define RATIONAL_BINOP_2(name,exp) \
    static PyObject* \
    pyrational_##name(PyObject* a, PyObject* b) { \
        rational x, y, z; \
        AS_RATIONAL(x,a); \
        AS_RATIONAL(y,b); \
        z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        PyTypeObject *type = rational_result_type(a, b); \
        return type == NULL ? NULL : PyRational_FromRational(type, z); \
    }
#define RATIONAL_BINOP(name) RATIONAL_BINOP_2(name,rational_##name(x,y))
RATIONAL_BINOP(add)
RATIONAL_BINOP(subtract)
RATIONAL_BINOP(multiply)
RATIONAL_BINOP(divide)
RATIONAL_BINOP(remainder)
RATIONAL_BINOP_2(floor_divide,
    make_rational_int(rational_floor(rational_divide(x,y))))

#define RATIONAL_UNOP(name,type,exp,convert) \
    static PyObject* \
    pyrational_##name(PyObject* self) { \
        rational x = ((PyRational*)self)->r; \
        type y = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return convert(y); \
    }
#define RATIONAL_UNOP_RAT(name,exp) \
    static PyObject* \
    pyrational_##name(PyObject* self) { \
        rational x = ((PyRational*)self)->r; \
        rational y = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        PyTypeObject *type = rational_result_type(self, self); \
        return type == NULL ? NULL : PyRational_FromRational(type, y); \
    }
RATIONAL_UNOP_RAT(negative,rational_negative(x))
RATIONAL_UNOP_RAT(absolute,rational_abs(x))
RATIONAL_UNOP(int,long,rational_int(x),PyLong_FromLong)
RATIONAL_UNOP(float,double,rational_double(x),PyFloat_FromDouble)

static PyObject*
pyrational_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static int
pyrational_nonzero(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    return rational_nonzero(x);
}

static PyObject*
pyrational_n(PyObject* self, void* closure) {
    return PyLong_FromLong(((PyRational*)self)->r.n);
}

static PyObject*
pyrational_d(PyObject* self, void* closure) {
    return PyLong_FromLong(d(((PyRational*)self)->r));
}

static PyGetSetDef pyrational_getset[] = {
    {(char*)"n",pyrational_n,0,(char*)"numerator",0},
    {(char*)"d",pyrational_d,0,(char*)"denominator",0},
    {0} /* sentinel */
};

static int
rational_scalar_traverse(PyObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}
static void
rational_scalar_dealloc(PyObject *self)
{
    PyTypeObject *type = Py_TYPE(self);
    PyObject_GC_UnTrack(self);
    type->tp_free(self);
    Py_DECREF(type);
}
static PyTypeObject *
create_rational_scalar(PyObject *module, PyTypeObject *base, int second)
{
    PyType_Slot slots[] = {
        {Py_tp_dealloc, rational_scalar_dealloc},
        {Py_tp_alloc, PyType_GenericAlloc},
        {Py_tp_free, PyObject_GC_Del},
        {Py_tp_traverse, rational_scalar_traverse},
        {Py_tp_doc, "Fixed precision rational numbers"},
        {Py_tp_new, pyrational_new},
        {Py_tp_repr, pyrational_repr},
        {Py_tp_str, pyrational_str},
        {Py_tp_hash, pyrational_hash},
        {Py_tp_richcompare, pyrational_richcompare},
        {Py_tp_getset, pyrational_getset},
        {Py_nb_add, pyrational_add},
        {Py_nb_subtract, pyrational_subtract},
        {Py_nb_multiply, pyrational_multiply},
        {Py_nb_remainder, pyrational_remainder},
        {Py_nb_negative, pyrational_negative},
        {Py_nb_positive, pyrational_positive},
        {Py_nb_absolute, pyrational_absolute},
        {Py_nb_bool, pyrational_nonzero},
        {Py_nb_int, pyrational_int},
        {Py_nb_float, pyrational_float},
        {Py_nb_floor_divide, pyrational_floor_divide},
        {Py_nb_true_divide, pyrational_divide},
        {0, NULL}
    };
    PyType_Spec spec = {
        .name = second ? "numpy._core._rational_tests.rational2"
                       : "numpy._core._rational_tests.rational",
        .basicsize = sizeof(PyRational),
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
        .slots = slots,
    };
    PyObject *bases = PyTuple_Pack(1, (PyObject *)base);
    if (bases == NULL) {
        return NULL;
    }
    PyObject *type = PyType_FromModuleAndSpec(module, &spec, bases);
    Py_DECREF(bases);
    return (PyTypeObject *)type;
}

/* NumPy support */

static PyObject*
npyrational_getitem(void* data, void* arr) {
    rational r;
    memcpy(&r,data,sizeof(rational));
    rational_module_state *state = rational_registered_state();
    return state == NULL ? NULL : PyRational_FromRational(state->rational, r);
}

static int
npyrational_setitem(PyObject* item, void* data, void* arr) {
    rational r;
    if (PyRational_Check(item)) {
        r = ((PyRational*)item)->r;
    }
    else {
        long long n = PyLong_AsLongLong(item);
        PyObject* y;
        int eq;
        if (error_converting(n)) {
            return -1;
        }
        y = PyLong_FromLongLong(n);
        if (!y) {
            return -1;
        }
        eq = PyObject_RichCompareBool(item, y, Py_EQ);
        Py_DECREF(y);
        if (eq<0) {
            return -1;
        }
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected rational, got %s", item->ob_type->tp_name);
            return -1;
        }
        r = make_rational_int(n);
    }
    memcpy(data, &r, sizeof(rational));
    return 0;
}

static inline void
byteswap(npy_int32* x) {
    char* p = (char*)x;
    size_t i;
    for (i = 0; i < sizeof(*x)/2; i++) {
        size_t j = sizeof(*x)-1-i;
        char t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
}

static void
npyrational_copyswapn(void* dst_, npy_intp dstride, void* src_,
        npy_intp sstride, npy_intp n, int swap, void* arr) {
    char *dst = (char*)dst_, *src = (char*)src_;
    npy_intp i;
    if (!src) {
        return;
    }
    if (swap) {
        for (i = 0; i < n; i++) {
            rational* r = (rational*)(dst+dstride*i);
            memcpy(r,src+sstride*i,sizeof(rational));
            byteswap(&r->n);
            byteswap(&r->dmm);
        }
    }
    else if (dstride == sizeof(rational) && sstride == sizeof(rational)) {
        memcpy(dst, src, n*sizeof(rational));
    }
    else {
        for (i = 0; i < n; i++) {
            memcpy(dst + dstride*i, src + sstride*i, sizeof(rational));
        }
    }
}

static void
npyrational_copyswap(void* dst, void* src, int swap, void* arr) {
    rational* r;
    if (!src) {
        return;
    }
    r = (rational*)dst;
    memcpy(r,src,sizeof(rational));
    if (swap) {
        byteswap(&r->n);
        byteswap(&r->dmm);
    }
}

static int
npyrational_compare(const void* d0, const void* d1, void* arr) {
    rational x = *(rational*)d0,
             y = *(rational*)d1;
    return rational_lt(x,y)?-1:rational_eq(x,y)?0:1;
}

#define FIND_EXTREME(name,op) \
    static int \
    npyrational_##name(void* data_, npy_intp n, \
            npy_intp* max_ind, void* arr) { \
        const rational* data; \
        npy_intp best_i; \
        rational best_r; \
        npy_intp i; \
        if (!n) { \
            return 0; \
        } \
        data = (rational*)data_; \
        best_i = 0; \
        best_r = data[0]; \
        for (i = 1; i < n; i++) { \
            if (rational_##op(data[i],best_r)) { \
                best_i = i; \
                best_r = data[i]; \
            } \
        } \
        *max_ind = best_i; \
        return 0; \
    }
FIND_EXTREME(argmin,lt)
FIND_EXTREME(argmax,gt)

static void
npyrational_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
        void* op, npy_intp n, void* arr) {
    rational r = {0};
    const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
    npy_intp i;
    for (i = 0; i < n; i++) {
        r = rational_add(r,rational_multiply(*(rational*)ip0,*(rational*)ip1));
        ip0 += is0;
        ip1 += is1;
    }
    *(rational*)op = r;
}

static npy_bool
npyrational_nonzero(void* data, void* arr) {
    rational r;
    memcpy(&r,data,sizeof(r));
    return rational_nonzero(r)?NPY_TRUE:NPY_FALSE;
}

static int
npyrational_fill(void* data_, npy_intp length, void* arr) {
    rational* data = (rational*)data_;
    rational delta = rational_subtract(data[1],data[0]);
    rational r = data[1];
    npy_intp i;
    for (i = 2; i < length; i++) {
        r = rational_add(r,delta);
        data[i] = r;
    }
    return 0;
}

static int
npyrational_fillwithscalar(void* buffer_, npy_intp length,
        void* value, void* arr) {
    rational r = *(rational*)value;
    rational* buffer = (rational*)buffer_;
    npy_intp i;
    for (i = 0; i < length; i++) {
        buffer[i] = r;
    }
    return 0;
}

typedef struct { char c; rational r; } align_test;

/*
 * A second rational variant registered via `PyArrayInitDTypeMeta_FromSpec`
 * with `NPY_DT_legacy_descriptor_proto`.  Same data layout as `rational`
 * (Python subclass of `PyRational_Type`), so all the existing C helpers,
 * casts, and ufunc loops can be reused for the new typenum/descr.
 */
static PyObject *
rational2_repr(PyObject *self)
{
    rational_module_state *state = PyType_GetModuleState(Py_TYPE(self));
    if (state == NULL) {
        return NULL;
    }
    rational_module_state *previous = rational_active;
    rational_active = state;
    PyObject *result = PyArrayDescr_Type.tp_repr(self);
    rational_active = previous;
    return result;
}
static PyArray_DTypeMeta *
create_rational_dtype(PyObject *module)
{
    PyType_Slot slots[] = {{Py_tp_repr, rational2_repr}, {0, NULL}};
    PyType_Spec spec = {
        .name = "numpy._core._rational_tests.Rational2DType",
        .basicsize = sizeof(_PyArray_LegacyDescr),
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
        .slots = slots,
    };
    PyObject *bases = PyTuple_Pack(1, (PyObject *)&PyArrayDescr_Type);
    if (bases == NULL) {
        return NULL;
    }
    PyObject *type = PyType_FromMetaclass(&PyArrayDTypeMeta_Type, module, &spec, bases);
    Py_DECREF(bases);
    return (PyArray_DTypeMeta *)type;
}

/*
 * Within-DType identity cast (memcpy).  Required by
 * `PyArrayInitDTypeMeta_FromSpec` since it rejects a NULL `casts`; for
 * full legacy registrations this is otherwise built lazily from `copyswap`.
 */
static int
rational2_within_dtype_loop(
        PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    char *src = data[0], *dst = data[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp N = dimensions[0];
    for (npy_intp i = 0; i < N; i++) {
        memcpy(dst, src, sizeof(rational));
        src += src_stride;
        dst += dst_stride;
    }
    return 0;
}

/*
 * `common_dtype` slot offered only by the new DType API.  Promotes
 * `rational2` with itself or any integer dtype to `rational2`; everything
 * else returns `Py_NotImplemented`.
 */
static PyArray_DTypeMeta *
rational2_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (cls == other) {
        Py_INCREF(cls);
        return cls;
    }
    int t = other->type_num;
    if (t == NPY_BOOL
            || t == NPY_BYTE || t == NPY_UBYTE
            || t == NPY_SHORT || t == NPY_USHORT
            || t == NPY_INT || t == NPY_UINT
            || t == NPY_LONG || t == NPY_ULONG
            || t == NPY_LONGLONG || t == NPY_ULONGLONG) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

/*
 * Minimal new-style getitem/setitem required by PyArrayInitDTypeMeta_FromSpec.
 * Reuse the legacy rational conversion logic for now.
 */
static PyObject *
rational2_getitem(PyArray_Descr *NPY_UNUSED(descr), char *data)
{
    return npyrational_getitem(data, NULL);
}

static int
rational2_setitem(
        PyArray_Descr *NPY_UNUSED(descr), PyObject *item, char *data)
{
    return npyrational_setitem(item, data, NULL);
}

#define DEFINE_CAST(From,To,statement) \
    static void \
    npycast_##From##_##To(void* from_, void* to_, npy_intp n, \
                          void* fromarr, void* toarr) { \
        const From* from = (From*)from_; \
        To* to = (To*)to_; \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            From x = from[i]; \
            statement \
            to[i] = y; \
        } \
    }
#define DEFINE_INT_CAST(bits) \
    DEFINE_CAST(npy_int##bits,rational,rational y = make_rational_int(x);) \
    DEFINE_CAST(rational,npy_int##bits,npy_int32 z = rational_int(x); \
                npy_int##bits y = z; if (y != z) set_overflow();)
DEFINE_INT_CAST(8)
DEFINE_INT_CAST(16)
DEFINE_INT_CAST(32)
DEFINE_INT_CAST(64)
DEFINE_CAST(rational,float,double y = rational_double(x);)
DEFINE_CAST(rational,double,double y = rational_double(x);)
DEFINE_CAST(npy_bool,rational,rational y = make_rational_int(x);)
DEFINE_CAST(rational,npy_bool,npy_bool y = rational_nonzero(x);)

#define BINARY_UFUNC(name,intype0,intype1,outtype,exp) \
    void name(char** args, npy_intp const *dimensions, \
              npy_intp const *steps, void* data) { \
        npy_intp is0 = steps[0], is1 = steps[1], \
            os = steps[2], n = *dimensions; \
        char *i0 = args[0], *i1 = args[1], *o = args[2]; \
        int k; \
        for (k = 0; k < n; k++) { \
            intype0 x = *(intype0*)i0; \
            intype1 y = *(intype1*)i1; \
            *(outtype*)o = exp; \
            i0 += is0; i1 += is1; o += os; \
        } \
    }
#define RATIONAL_BINARY_UFUNC(name,type,exp) \
    BINARY_UFUNC(rational_ufunc_##name,rational,rational,type,exp)
RATIONAL_BINARY_UFUNC(add,rational,rational_add(x,y))
RATIONAL_BINARY_UFUNC(subtract,rational,rational_subtract(x,y))
RATIONAL_BINARY_UFUNC(multiply,rational,rational_multiply(x,y))
RATIONAL_BINARY_UFUNC(divide,rational,rational_divide(x,y))
RATIONAL_BINARY_UFUNC(remainder,rational,rational_remainder(x,y))
RATIONAL_BINARY_UFUNC(floor_divide,rational,
    make_rational_int(rational_floor(rational_divide(x,y))))
PyUFuncGenericFunction rational_ufunc_true_divide = rational_ufunc_divide;
RATIONAL_BINARY_UFUNC(minimum,rational,rational_lt(x,y)?x:y)
RATIONAL_BINARY_UFUNC(maximum,rational,rational_lt(x,y)?y:x)
RATIONAL_BINARY_UFUNC(equal,npy_bool,rational_eq(x,y))
RATIONAL_BINARY_UFUNC(not_equal,npy_bool,rational_ne(x,y))
RATIONAL_BINARY_UFUNC(less,npy_bool,rational_lt(x,y))
RATIONAL_BINARY_UFUNC(greater,npy_bool,rational_gt(x,y))
RATIONAL_BINARY_UFUNC(less_equal,npy_bool,rational_le(x,y))
RATIONAL_BINARY_UFUNC(greater_equal,npy_bool,rational_ge(x,y))

BINARY_UFUNC(gcd_ufunc,npy_int64,npy_int64,npy_int64,gcd(x,y))
BINARY_UFUNC(lcm_ufunc,npy_int64,npy_int64,npy_int64,lcm(x,y))

#define UNARY_UFUNC(name,type,exp) \
    void rational_ufunc_##name(char** args, npy_intp const *dimensions, \
                               npy_intp const *steps, void* data) { \
        npy_intp is = steps[0], os = steps[1], n = *dimensions; \
        char *i = args[0], *o = args[1]; \
        int k; \
        for (k = 0; k < n; k++) { \
            rational x = *(rational*)i; \
            *(type*)o = exp; \
            i += is; o += os; \
        } \
    }
UNARY_UFUNC(negative,rational,rational_negative(x))
UNARY_UFUNC(absolute,rational,rational_abs(x))
UNARY_UFUNC(floor,rational,make_rational_int(rational_floor(x)))
UNARY_UFUNC(ceil,rational,make_rational_int(rational_ceil(x)))
UNARY_UFUNC(trunc,rational,make_rational_int(x.n/d(x)))
UNARY_UFUNC(square,rational,rational_multiply(x,x))
UNARY_UFUNC(rint,rational,make_rational_int(rational_rint(x)))
UNARY_UFUNC(sign,rational,make_rational_int(rational_sign(x)))
UNARY_UFUNC(reciprocal,rational,rational_inverse(x))
UNARY_UFUNC(numerator,npy_int64,x.n)
UNARY_UFUNC(denominator,npy_int64,d(x))

static inline void
rational_matrix_multiply(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    /* pointers to data for input and output arrays */
    char *ip1 = args[0];
    char *ip2 = args[1];
    char *op = args[2];

    /* lengths of core dimensions */
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    /* striding over core dimensions */
    npy_intp is1_m = steps[0];
    npy_intp is1_n = steps[1];
    npy_intp is2_n = steps[2];
    npy_intp is2_p = steps[3];
    npy_intp os_m = steps[4];
    npy_intp os_p = steps[5];

    /* core dimensions counters */
    npy_intp m, p;

    /* calculate dot product for each row/column vector pair */
    for (m = 0; m < dm; m++) {
        for (p = 0; p < dp; p++) {
            npyrational_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

            /* advance to next column of 2nd input array and output array */
            ip2 += is2_p;
            op  +=  os_p;
        }

        /* reset to first column of 2nd input array and output array */
        ip2 -= is2_p * p;
        op -= os_p * p;

        /* advance to next row of 1st input array and output array */
        ip1 += is1_m;
        op += os_m;
    }
}


static void
rational_gufunc_matrix_multiply(char **args, npy_intp const *dimensions,
                                npy_intp const *steps, void *NPY_UNUSED(func))
{
    /* outer dimensions counter */
    npy_intp N_;

    /* length of flattened outer dimensions */
    npy_intp dN = dimensions[0];

    /* striding over flattened outer dimensions for input and output arrays */
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];

    /*
     * loop through outer dimensions, performing matrix multiply on
     * core dimensions for each loop
     */
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
        rational_matrix_multiply(args, dimensions+1, steps+3);
    }
}


static void
rational_ufunc_test_add(char** args, npy_intp const *dimensions,
                        npy_intp const *steps, void* data) {
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];
    int k;
    for (k = 0; k < n; k++) {
        npy_int64 x = *(npy_int64*)i0;
        npy_int64 y = *(npy_int64*)i1;
        *(rational*)o = rational_add(make_rational_fast(x, 1),
                                     make_rational_fast(y, 1));
        i0 += is0; i1 += is1; o += os;
    }
}


static void
rational_ufunc_test_add_rationals(char** args, npy_intp const *dimensions,
                        npy_intp const *steps, void* data) {
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];
    int k;
    for (k = 0; k < n; k++) {
        rational x = *(rational*)i0;
        rational y = *(rational*)i1;
        *(rational*)o = rational_add(x, y);
        i0 += is0; i1 += is1; o += os;
    }
}


/* Register the rational <-> builtin casts and ufunc loops for `typenum`.
 * Called once per variant since the data layout is identical. */
static int
register_rational_casts_and_ufuncs(int typenum,
                                   PyArray_Descr *descr,
                                   PyObject *numpy)
{
    #define REGISTER_CAST(From,To,from_descr,to_typenum,safe) { \
            PyArray_Descr *owned_descr = (PyArray_Descr *)(from_descr); \
            if (owned_descr == NULL) { \
                return -1; \
            } \
            int result = PyArray_RegisterCastFunc(owned_descr, \
                    (to_typenum), npycast_##From##_##To); \
            if (result == 0 && safe) { \
                result = PyArray_RegisterCanCast(owned_descr, \
                        (to_typenum), NPY_NOSCALAR); \
            } \
            Py_DECREF(owned_descr); \
            if (result < 0) { \
                return -1; \
            } \
        }
    #define REGISTER_INT_CASTS(bits) \
        REGISTER_CAST(npy_int##bits, rational, \
                      PyArray_DescrFromType(NPY_INT##bits), typenum, 1) \
        REGISTER_CAST(rational, npy_int##bits, (PyArray_Descr *)Py_NewRef(descr), NPY_INT##bits, 0)
    REGISTER_INT_CASTS(8)
    REGISTER_INT_CASTS(16)
    REGISTER_INT_CASTS(32)
    REGISTER_INT_CASTS(64)
    REGISTER_CAST(rational, float, (PyArray_Descr *)Py_NewRef(descr), NPY_FLOAT, 0)
    REGISTER_CAST(rational, double, (PyArray_Descr *)Py_NewRef(descr), NPY_DOUBLE, 1)
    REGISTER_CAST(npy_bool, rational,
                  PyArray_DescrFromType(NPY_BOOL), typenum, 1)
    REGISTER_CAST(rational, npy_bool, (PyArray_Descr *)Py_NewRef(descr), NPY_BOOL, 0)
    #undef REGISTER_CAST
    #undef REGISTER_INT_CASTS

    #define REGISTER_UFUNC(name,...) { \
        PyUFuncObject* ufunc = \
            (PyUFuncObject*)PyObject_GetAttrString(numpy, #name); \
        int _types[] = __VA_ARGS__; \
        if (!ufunc) { \
            return -1; \
        } \
        if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
            PyErr_Format(PyExc_AssertionError, \
                         "ufunc %s takes %d arguments, our loop takes %lu", \
                         #name, ufunc->nargs, (unsigned long) \
                         (sizeof(_types)/sizeof(int))); \
            Py_DECREF(ufunc); \
            return -1; \
        } \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, typenum, \
                rational_ufunc_##name, _types, 0) < 0) { \
            Py_DECREF(ufunc); \
            return -1; \
        } \
        Py_DECREF(ufunc); \
    }
    #define REGISTER_UFUNC_BINARY_RATIONAL(name) \
        REGISTER_UFUNC(name, {typenum, typenum, typenum})
    #define REGISTER_UFUNC_BINARY_COMPARE(name) \
        REGISTER_UFUNC(name, {typenum, typenum, NPY_BOOL})
    #define REGISTER_UFUNC_UNARY(name) \
        REGISTER_UFUNC(name, {typenum, typenum})
    /* Binary */
    REGISTER_UFUNC_BINARY_RATIONAL(add)
    REGISTER_UFUNC_BINARY_RATIONAL(subtract)
    REGISTER_UFUNC_BINARY_RATIONAL(multiply)
    REGISTER_UFUNC_BINARY_RATIONAL(divide)
    REGISTER_UFUNC_BINARY_RATIONAL(remainder)
    REGISTER_UFUNC_BINARY_RATIONAL(true_divide)
    REGISTER_UFUNC_BINARY_RATIONAL(floor_divide)
    REGISTER_UFUNC_BINARY_RATIONAL(minimum)
    REGISTER_UFUNC_BINARY_RATIONAL(maximum)
    /* Comparisons */
    REGISTER_UFUNC_BINARY_COMPARE(equal)
    REGISTER_UFUNC_BINARY_COMPARE(not_equal)
    REGISTER_UFUNC_BINARY_COMPARE(less)
    REGISTER_UFUNC_BINARY_COMPARE(greater)
    REGISTER_UFUNC_BINARY_COMPARE(less_equal)
    REGISTER_UFUNC_BINARY_COMPARE(greater_equal)
    /* Unary */
    REGISTER_UFUNC_UNARY(negative)
    REGISTER_UFUNC_UNARY(absolute)
    REGISTER_UFUNC_UNARY(floor)
    REGISTER_UFUNC_UNARY(ceil)
    REGISTER_UFUNC_UNARY(trunc)
    REGISTER_UFUNC_UNARY(rint)
    REGISTER_UFUNC_UNARY(square)
    REGISTER_UFUNC_UNARY(reciprocal)
    REGISTER_UFUNC_UNARY(sign)
    #undef REGISTER_UFUNC
    #undef REGISTER_UFUNC_BINARY_RATIONAL
    #undef REGISTER_UFUNC_BINARY_COMPARE
    #undef REGISTER_UFUNC_UNARY

    return 0;
}


static PyMethodDef module_methods[] = {
    {0} /* sentinel */
};

static int
rational_add_owned(PyObject *module, const char *name, PyObject *value)
{
    if (PyModule_AddObject(module, name, value) < 0) {
        Py_DECREF(value);
        return -1;
    }
    return 0;
}

static int
initialize_rational_tests(PyObject *m)
{
    PyObject *numpy = NULL;
    int npy_rational, npy_rational2;
    PyArray_Descr *npyrational_descr = NULL;
    PyArray_Descr *npyrational2_descr = NULL;
    PyArray_ArrFuncs npyrational_arrfuncs;
    numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        goto fail;
    }
    rational_active->rational = create_rational_scalar(m, &PyGenericArrType_Type, 0);
    if (rational_active->rational == NULL) {
        goto fail;
    }
    PyArray_DescrProto npyrational_descr_proto = {
    PyObject_HEAD_INIT(0)
    &PyRational_Type,       /* typeobj */
    'V',                    /* kind */
    'r',                    /* type */
    '=',                    /* byteorder */
    /*
     * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
     * exceptions.  This isn't technically necessary,
     * since we're careful about thread safety, and hopefully future
     * versions of numpy will recognize that.
     */
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* hasobject */
    0,                      /* type_num */
    sizeof(rational),       /* elsize */
    offsetof(align_test,r), /* alignment */
    0,                      /* subarray */
    0,                      /* fields */
    0,                      /* names */
    &npyrational_arrfuncs,  /* f */
};


    /* Initialize rational descriptor */
    PyArray_InitArrFuncs(&npyrational_arrfuncs);
    npyrational_arrfuncs.getitem = npyrational_getitem;
    npyrational_arrfuncs.setitem = npyrational_setitem;
    npyrational_arrfuncs.copyswapn = npyrational_copyswapn;
    npyrational_arrfuncs.copyswap = npyrational_copyswap;
    npyrational_arrfuncs.compare = npyrational_compare;
    npyrational_arrfuncs.argmin = npyrational_argmin;
    npyrational_arrfuncs.argmax = npyrational_argmax;
    npyrational_arrfuncs.dotfunc = npyrational_dot;
    npyrational_arrfuncs.nonzero = npyrational_nonzero;
    npyrational_arrfuncs.fill = npyrational_fill;
    npyrational_arrfuncs.fillwithscalar = npyrational_fillwithscalar;
    /* Left undefined: scanfunc, fromstr, sort, argsort */
    Py_SET_TYPE(&npyrational_descr_proto, &PyArrayDescr_Type);
    npy_rational = PyArray_RegisterDataType(&npyrational_descr_proto);
    if (npy_rational<0) {
        goto fail;
    }
    npyrational_descr = PyArray_DescrFromType(npy_rational);
    if (npyrational_descr == NULL) {
        goto fail;
    }

    /* Support dtype(rational) syntax */
    if (PyDict_SetItemString(PyRational_Type.tp_dict, "dtype",
                             (PyObject*)npyrational_descr) < 0) {
        goto fail;
    }

    /* Register casts and ufunc loops for the legacy `rational` dtype. */
    if (register_rational_casts_and_ufuncs(
            npy_rational, npyrational_descr, numpy) < 0) {
        goto fail;
    }

    /*
     * Register a second rational dtype through the new DType API using the
     * `NPY_DT_legacy_descriptor_proto` slot.  We reuse the proto above
     * (already copied out by `PyArray_RegisterDataType`) and only patch
     * the fields that must differ for the new variant.
     *
     * NOTE(seberg): Once parts of the legacy API are deprecated (e.g.
     * adding casts) we can clean this up to only use the rational2 path.
     */
    rational_active->rational2 = create_rational_scalar(m, &PyRational_Type, 1);
    if (rational_active->rational2 == NULL) {
        goto fail;
    }
    rational_active->dtype2 = (PyTypeObject *)create_rational_dtype(m);
    if (rational_active->dtype2 == NULL) {
        goto fail;
    }

    npyrational_descr_proto.typeobj = &PyRational2_Type;
    npyrational_descr_proto.type = 'R';
    npyrational_descr_proto.type_num = 0;

    {
        PyType_Slot dtype_slots[] = {
            {NPY_DT_legacy_descriptor_proto, &npyrational_descr_proto},
            {NPY_DT_common_dtype, rational2_common_dtype},
            {NPY_DT_getitem, rational2_getitem},
            {NPY_DT_setitem, rational2_setitem},
            {0, NULL},
        };

        /* Within-DType identity cast (NULL `dtypes` slots are filled in
         * by `PyArrayInitDTypeMeta_FromSpec`). */
        PyType_Slot within_dtype_slots[] = {
            {NPY_METH_strided_loop, rational2_within_dtype_loop},
            {NPY_METH_unaligned_strided_loop, rational2_within_dtype_loop},
            {0, NULL},
        };
        PyArray_DTypeMeta *within_dtype_dtypes[2] = {NULL, NULL};
        PyArrayMethod_Spec within_dtype_cast = {
            .name = "cast_rational2_within_dtype",
            .nin = 1,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = within_dtype_dtypes,
            .slots = within_dtype_slots,
        };
        PyArrayMethod_Spec *casts[] = {&within_dtype_cast, NULL};

        PyArrayDTypeMeta_Spec dtype_spec = {
            .typeobj = &PyRational2_Type,
            .flags = 0,
            .casts = casts,
            .slots = dtype_slots,
            .baseclass = NULL,
        };

        if (PyArrayInitDTypeMeta_FromSpec(
                &NPY_Rational2DType, &dtype_spec) < 0) {
            goto fail;
        }
    }
    npy_rational2 = NPY_Rational2DType.type_num;
    npyrational2_descr = NPY_Rational2DType.singleton;
    Py_INCREF(npyrational2_descr);

    /* Support `dtype(rational2)` */
    if (PyDict_SetItemString(PyRational2_Type.tp_dict, "dtype",
            (PyObject *)npyrational2_descr) < 0) {
        goto fail;
    }

    if (register_rational_casts_and_ufuncs(
            npy_rational2, npyrational2_descr, numpy) < 0) {
        goto fail;
    }

    PyType_Modified(&PyRational_Type);
    PyType_Modified(&PyRational2_Type);
#if PY_VERSION_HEX >= 0x030e0000
    if (PyType_Freeze(&PyRational_Type) < 0 || PyType_Freeze(&PyRational2_Type) < 0) {
        goto fail;
    }
#endif
    if (PyModule_AddObjectRef(m, "rational", (PyObject *)&PyRational_Type) < 0
            || PyModule_AddObjectRef(m, "rational2", (PyObject *)&PyRational2_Type) < 0
            || PyModule_AddObjectRef(m, "Rational2DType", (PyObject *)&NPY_Rational2DType) < 0) {
        goto fail;
    }

    /* Create matrix multiply generalized ufunc */
    {
        int types2[3] = {npy_rational,npy_rational,npy_rational};
        PyObject* gufunc = PyUFunc_FromFuncAndDataAndSignature(0,0,0,0,2,1,
            PyUFunc_None,"matrix_multiply",
            "return result of multiplying two matrices of rationals",
            0,"(m,n),(n,p)->(m,p)");
        if (!gufunc) {
            goto fail;
        }
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)gufunc, npy_rational,
                rational_gufunc_matrix_multiply, types2, 0) < 0) {
            Py_DECREF(gufunc);
            goto fail;
        }
        if (rational_add_owned(m,"matrix_multiply",(PyObject*)gufunc) < 0) {
            goto fail;
        }
    }

    /* Create test ufunc with built in input types and rational output type */
    {
        int types3[3] = {NPY_INT64,NPY_INT64,npy_rational};

        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,2,1,
                PyUFunc_None,"test_add",
                "add two matrices of int64 and return rational matrix",0);
        if (!ufunc) {
            goto fail;
        }
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, npy_rational,
                rational_ufunc_test_add, types3, 0) < 0) {
            Py_DECREF(ufunc);
            goto fail;
        }
        if (rational_add_owned(m,"test_add",(PyObject*)ufunc) < 0) {
            goto fail;
        }
    }

    /* Create test ufunc with rational types using RegisterLoopForDescr */
    {
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,2,1,
                PyUFunc_None,"test_add_rationals",
                "add two matrices of rationals and return rational matrix",0);
        PyArray_Descr* types[3] = {npyrational_descr,
                                    npyrational_descr,
                                    npyrational_descr};

        if (!ufunc) {
            goto fail;
        }
        if (PyUFunc_RegisterLoopForDescr((PyUFuncObject*)ufunc, npyrational_descr,
                rational_ufunc_test_add_rationals, types, 0) < 0) {
            Py_DECREF(ufunc);
            goto fail;
        }
        if (rational_add_owned(m,"test_add_rationals",(PyObject*)ufunc) < 0) {
            goto fail;
        }
    }

    /* Create numerator and denominator ufuncs */
    #define NEW_UNARY_UFUNC(name,type,doc) { \
        int types[2] = {npy_rational,type}; \
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,1,1, \
            PyUFunc_None,#name,doc,0); \
        if (!ufunc) { \
            goto fail; \
        } \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, \
                npy_rational,rational_ufunc_##name,types,0)<0) { \
            Py_DECREF(ufunc); \
            goto fail; \
        } \
        if (rational_add_owned(m,#name,(PyObject*)ufunc) < 0) { \
            goto fail; \
        } \
    }
    NEW_UNARY_UFUNC(numerator,NPY_INT64,"rational number numerator");
    NEW_UNARY_UFUNC(denominator,NPY_INT64,"rational number denominator");

    /* Create gcd and lcm ufuncs */
    #define GCD_LCM_UFUNC(name,type,doc) { \
        static const PyUFuncGenericFunction func[1] = {name##_ufunc}; \
        static const char types[3] = {type,type,type}; \
        static void* data[1] = {0}; \
        PyObject* ufunc = PyUFunc_FromFuncAndData( \
            (PyUFuncGenericFunction*)func, data,types, \
            1,2,1,PyUFunc_One,#name,doc,0); \
        if (!ufunc) { \
            goto fail; \
        } \
        if (owned_legacy_test_install((PyUFuncObject *)ufunc) < 0) { \
            Py_DECREF(ufunc); \
            goto fail; \
        } \
        if (rational_add_owned(m,#name,(PyObject*)ufunc) < 0) { \
            goto fail; \
        } \
    }
    GCD_LCM_UFUNC(gcd,NPY_INT64,"greatest common denominator of two integers");
    GCD_LCM_UFUNC(lcm,NPY_INT64,"least common multiple of two integers");

    Py_DECREF(numpy);
    Py_DECREF(npyrational_descr);
    Py_DECREF(npyrational2_descr);
    return 0;

fail:
    Py_XDECREF(numpy);
    Py_XDECREF(npyrational_descr);
    Py_XDECREF(npyrational2_descr);
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "cannot load _rational_tests module");
    }
    return -1;
}

static int
rational_module_traverse(PyObject *module, visitproc visit, void *arg)
{
    rational_module_state *state = PyModule_GetState(module);
    if (state == NULL) {
        return 0;
    }
    Py_VISIT(state->rational);
    Py_VISIT(state->rational2);
    Py_VISIT(state->dtype2);
    Py_VISIT(state->shared_owner);
    return NPY_SI_Context_Traverse(&state->numpy, visit, arg);
}
static int
rational_module_clear(PyObject *module)
{
    rational_module_state *state = PyModule_GetState(module);
    if (state == NULL) {
        return 0;
    }
    Py_CLEAR(state->shared_owner);
    Py_CLEAR(state->rational);
    Py_CLEAR(state->rational2);
    Py_CLEAR(state->dtype2);
    NPY_SI_Context_Clear(&state->numpy);
    return 0;
}
static void
rational_module_free(void *module)
{
    rational_module_clear((PyObject *)module);
}
static int
rational_module_exec(PyObject *module)
{
    rational_module_state *state = PyModule_GetState(module);
    if (state->initialized) {
        return 0;
    }
    PyObject *canonical = rational_cached_module();
    if (canonical != NULL && canonical != module) {
        const char *names[] = {"rational", "rational2", "Rational2DType",
            "matrix_multiply", "test_add", "test_add_rationals", "numerator",
            "denominator", "gcd", "lcm", NULL};
        for (int i = 0; names[i] != NULL; i++) {
            PyObject *value = PyObject_GetAttrString(canonical, names[i]);
            if (value == NULL) {
                return -1;
            }
            int result = PyModule_AddObjectRef(module, names[i], value);
            Py_DECREF(value);
            if (result < 0) {
                return -1;
            }
        }
        state->shared_owner = Py_NewRef(canonical);
        state->initialized = 1;
        return 0;
    }
    if (PyErr_Occurred() || NPY_SI_Context_Import(&state->numpy) < 0) {
        return -1;
    }
    typedef unsigned int (*feature_version_func)(void);
    state->runtime_version = (int)((feature_version_func)state->numpy.array_api[211])();
    rational_module_state *previous = rational_active;
    rational_active = state;
    int result = initialize_rational_tests(module);
    rational_active = previous;
    if (result < 0) {
        return -1;
    }
    PyObject *dict = PyInterpreterState_GetDict(PyThreadState_GetInterpreter(PyThreadState_Get()));
    if (dict == NULL || PyDict_SetItemString(dict, rational_cache_key, module) < 0) {
        return -1;
    }
    state->initialized = 1;
    return 0;
}
static PyModuleDef_Slot rational_slots[] = {
    {Py_mod_exec, rational_module_exec},
#if PY_VERSION_HEX >= 0x030f0000
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#elif PY_VERSION_HEX >= 0x030c0000
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};
static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_rational_tests",
    .m_size = sizeof(rational_module_state),
    .m_methods = module_methods,
    .m_slots = rational_slots,
    .m_traverse = rational_module_traverse,
    .m_clear = rational_module_clear,
    .m_free = rational_module_free
};
PyMODINIT_FUNC PyInit__rational_tests(void)
{
    return PyModuleDef_Init(&moduledef);
}
