"""Regression checks for Python callbacks during native iterator cleanup."""
import gc
import subprocess
import sys
import weakref

import numpy as np
import pytest


@pytest.mark.parametrize("buffered", [False, True])
def test_operand_finalizer_can_reenter_close(buffered):
    # Returning from close alone is insufficient: double release can corrupt
    # allocator state and hang only when the process subsequently finalizes.
    code = r'''
import gc
import sys
import numpy as np

holder = {}
events = []
errors = []

class Operand:
    def __del__(self):
        try:
            iterator = holder['iterator']
            events.append('entered')
            iterator.close()
            try:
                iterator.__init__(np.array([1]))
            except ValueError:
                events.append('initialization-rejected-while-closing')
            else:
                errors.append('initialization during native cleanup was allowed')
            events.append('returned')
        except BaseException as error:
            errors.append(repr(error))

array = np.array([Operand()], dtype=object)
flags = ['refs_ok'] + (['buffered'] if sys.argv[1] == 'True' else [])
iterator = np.nditer(array, flags=flags, op_flags=['readonly'], buffersize=1)
holder['iterator'] = iterator
del array
iterator.close()
assert not errors, errors
assert events == ['entered', 'initialization-rejected-while-closing', 'returned'], events
iterator.close()
holder.clear()
del iterator
gc.collect()
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(buffered)],
                   check=True, timeout=30)


@pytest.mark.parametrize("buffered", [False, True])
def test_collect_cycle_through_operand(buffered):
    class OwnedArray(np.ndarray):
        pass

    array = np.arange(4).view(OwnedArray)
    flags = ["buffered"] if buffered else []
    iterator = np.nditer(array, flags=flags)
    assert iterator.operands[0] is array
    array.iterator = iterator
    reference = weakref.ref(array)
    del array
    gc.collect()
    assert reference() is iterator.operands[0]
    del iterator
    gc.collect()
    assert reference() is None
