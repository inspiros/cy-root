"""Native ownership, buffer-protocol lifetime, and stride boundary checks."""
import array
import ctypes
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import weakref

import numpy as np
from numpy.testing import assert_array_equal

import cyroot
from cyroot.native._buffers import DoubleBuffer


class NativeHeaderTests(unittest.TestCase):
    def test_standalone_header(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('A C++ compiler is not available for the standalone header check.')
        source = Path(__file__).resolve().parent / 'native' / 'test_buffers.cpp'
        include = Path(cyroot.__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='cyroot-native-buffers-') as temporary:
            executable = Path(temporary) / ('test_buffers.exe' if os.name == 'nt' else 'test_buffers')
            command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                       f'-I{include}', str(source), '-o', str(executable)]
            environment = os.environ.copy()
            if environment.get('CYROOT_NATIVE_SANITIZERS') == '1':
                command.extend(['-g', '-O1', '-fno-omit-frame-pointer', '-fsanitize=address,undefined'])
                # LeakSanitizer cannot run under some traced/sandboxed hosts.
                # The C++ fixture independently counts every owned allocation.
                environment.setdefault('ASAN_OPTIONS', 'detect_leaks=0:halt_on_error=1')
                environment.setdefault('UBSAN_OPTIONS', 'halt_on_error=1:print_stacktrace=1')
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True,
                           text=True, env=environment)


class DoubleBufferTests(unittest.TestCase):
    def test_copy_truthiness_is_evaluated_once_and_borrowing_stays_pinned(self):
        class ChangingFlag:
            calls = 0

            def __bool__(self):
                self.calls += 1
                return self.calls > 1

        flag = ChangingFlag()
        source = np.arange(3., dtype=np.float64)
        reference = weakref.ref(source)
        guard = DoubleBuffer(source, copy=flag)
        del source
        gc.collect()
        self.assertEqual(flag.calls, 1)
        self.assertIsNotNone(reference())
        self.assertFalse(guard.owns_data)
        self.assertEqual(guard.tolist(), [0., 1., 2.])

    def test_copy_conversion_cannot_reenter_initialization(self):
        guard = DoubleBuffer.__new__(DoubleBuffer)
        case = self

        class ReentrantFlag:
            def __bool__(self):
                with case.assertRaises(RuntimeError):
                    guard.__init__(array.array('d', [99.]), copy=True)
                return False

        guard.__init__(array.array('d', [1., 2.]), copy=ReentrantFlag())
        self.assertEqual(guard.tolist(), [1., 2.])

    def test_builtin_buffer_without_numpy_conversion(self):
        source = array.array('d', [1., 2., 3.])
        guard = DoubleBuffer(source)
        self.assertEqual(guard.shape, (3,))
        self.assertEqual(guard.strides, (8,))
        self.assertEqual(guard.tolist(), [1., 2., 3.])
        self.assertEqual(guard[-1], 3.)
        self.assertTrue(guard.readonly)
        self.assertFalse(guard.owns_data)
        source[0] = 7.
        self.assertEqual(guard[0], 7.)
        with self.assertRaises(TypeError):
            guard[0] = 10.
        with self.assertRaises(TypeError):
            guard.fill(0.)
        # The managed export prevents storage reallocation until the guard dies.
        with self.assertRaises(BufferError):
            source.append(4.)
        del guard
        gc.collect()
        source.append(4.)

    def test_solver_boundaries_convert_other_real_buffers_elementwise(self):
        residual = lambda p: (p[0]-1., p[1]-2.)
        self.assertTrue(cyroot.generalized_newton(residual, None, np.array([0, 0])).success)
        self.assertTrue(cyroot.generalized_newton(
            lambda p: np.array(residual(p), dtype=np.float32), None, (0., 0.)).success)
        self.assertEqual(cyroot.generalized_newton(
            residual, lambda p: np.eye(2, dtype=np.int64), array.array('i', [0, 0])).root, (1., 2.))
        self.assertTrue(cyroot.vector_levenberg_marquardt(residual, None, np.array([0, 0])).success)
        for complex_input in (lambda: cyroot.generalized_newton(residual, None, np.zeros(2, complex)),
                              lambda: cyroot.generalized_newton(
                                  lambda p: np.array(residual(p), dtype=complex), None, (0., 0.))):
            with self.assertRaisesRegex(TypeError, 'complex'):
                complex_input()

    def test_negative_and_zero_strides(self):
        source = np.arange(20., dtype=np.float64).reshape(4, 5)
        for values in (source[::-1, ::-2], source.T,
                       np.broadcast_to(np.array([1., 2., 3.]), (4, 3)),
                       source.reshape(-1)[::-3]):
            with self.subTest(shape=values.shape, strides=values.strides):
                guard = DoubleBuffer(values)
                self.assertEqual(guard.shape, values.shape)
                self.assertEqual(guard.strides, values.strides)
                self.assertEqual(guard.tolist(), values.tolist())
                export = memoryview(guard)
                self.assertEqual(export.shape, values.shape)
                self.assertEqual(export.strides, values.strides)
                self.assertTrue(export.readonly)
                self.assertEqual(export.tolist(), values.tolist())
                self.assertEqual(export.tobytes(), values.tobytes(order='C'))

    def test_readonly_source_can_be_borrowed_or_privately_copied(self):
        source = np.arange(6., dtype=np.float64).reshape(2, 3)
        source.setflags(write=False)
        borrowed = DoubleBuffer(source)
        private = DoubleBuffer(source, copy=True)
        self.assertTrue(borrowed.readonly)
        self.assertFalse(private.readonly)
        self.assertTrue(private.owns_data)
        self.assertEqual(private.strides, (24, 8))
        private[0, 1] = 99.
        self.assertEqual(private[0, 1], 99.)
        self.assertEqual(source[0, 1], 1.)
        export = memoryview(private)
        self.assertFalse(export.readonly)
        export[1, 2] = -5.
        self.assertEqual(private[1, 2], -5.)
        self.assertEqual(source[1, 2], 5.)
        private.fill(2.5)
        assert_array_equal(np.asarray(private), np.full((2, 3), 2.5))

    def test_copy_preserves_values_and_becomes_contiguous(self):
        source = np.arange(18., dtype=np.float64).reshape(3, 6)[::-1, ::-2]
        guard = DoubleBuffer(source)
        private = guard.copy()
        self.assertEqual(private.tolist(), source.tolist())
        self.assertEqual(private.strides, (24, 8))
        self.assertTrue(memoryview(private).c_contiguous)
        private[0, 0] = -100.
        self.assertNotEqual(source[0, 0], -100.)
        self.assertEqual(guard.tolist(), source.tolist())

    def test_borrowed_export_is_readonly_even_for_mutable_source(self):
        source = np.arange(4., dtype=np.float64)
        guard = DoubleBuffer(source)
        export = memoryview(guard)
        self.assertTrue(export.readonly)
        with self.assertRaises(TypeError):
            export[0] = 7.
        converted = np.asarray(guard)
        self.assertFalse(converted.flags.writeable)
        self.assertTrue(np.shares_memory(source, converted))

    def test_format_request_without_shape_reports_double_itemsize(self):
        # PyBUF_FORMAT can be requested independently of PyBUF_ND.
        class BufferInfo(ctypes.Structure):
            _fields_ = [
                ('buf', ctypes.c_void_p), ('obj', ctypes.c_void_p),
                ('length', ctypes.c_ssize_t), ('itemsize', ctypes.c_ssize_t),
                ('readonly', ctypes.c_int), ('ndim', ctypes.c_int),
                ('format', ctypes.c_char_p), ('shape', ctypes.POINTER(ctypes.c_ssize_t)),
                ('strides', ctypes.POINTER(ctypes.c_ssize_t)),
                ('suboffsets', ctypes.POINTER(ctypes.c_ssize_t)), ('internal', ctypes.c_void_p),
            ]

        get_buffer = ctypes.PYFUNCTYPE(
            ctypes.c_int, ctypes.py_object, ctypes.POINTER(BufferInfo), ctypes.c_int
        )(('PyObject_GetBuffer', ctypes.pythonapi))
        release_buffer = ctypes.PYFUNCTYPE(None, ctypes.POINTER(BufferInfo))(
            ('PyBuffer_Release', ctypes.pythonapi))
        info = BufferInfo()
        guard = DoubleBuffer(array.array('d', [1., 2., 3.]))
        get_buffer(guard, ctypes.byref(info), 0x0004)  # PyBUF_FORMAT
        try:
            self.assertEqual(info.itemsize, 8)
            self.assertEqual(info.format, b'd')
            self.assertEqual(info.ndim, 1)
            self.assertEqual(info.length, 24)
            self.assertFalse(info.shape)
        finally:
            release_buffer(ctypes.byref(info))

    def test_contiguous_requests_reject_strided_borrows(self):
        guard = DoubleBuffer(np.arange(8., dtype=np.float64)[::2])
        with self.assertRaises(BufferError):
            np.frombuffer(guard, dtype=np.float64)
        private = guard.copy()
        assert_array_equal(np.frombuffer(private, dtype=np.float64), [0., 2., 4., 6.])

    def test_owned_export_retains_native_allocation(self):
        private = DoubleBuffer(array.array('d', [1., 2., 3.]), copy=True)
        export = memoryview(private)
        del private
        gc.collect()
        self.assertEqual(export.tolist(), [1., 2., 3.])
        export[1] = -2.
        self.assertEqual(export.tolist(), [1., -2., 3.])
        export.release()

    def test_exporter_and_guard_lifetimes(self):
        source = np.arange(4., dtype=np.float64)
        source_reference = weakref.ref(source)
        guard = DoubleBuffer(source)
        export = memoryview(guard)
        del source, guard
        gc.collect()
        self.assertIsNotNone(source_reference())
        self.assertEqual(export.tolist(), [0., 1., 2., 3.])
        export.release()
        gc.collect()
        self.assertIsNone(source_reference())

    def test_private_copy_releases_exporter_immediately(self):
        source = np.arange(4., dtype=np.float64)
        source_reference = weakref.ref(source)
        private = DoubleBuffer(source, copy=True)
        del source
        gc.collect()
        self.assertIsNone(source_reference())
        self.assertEqual(private.tolist(), [0., 1., 2., 3.])

    def test_managed_export_is_visible_to_cyclic_gc(self):
        class CyclicArray(np.ndarray):
            pass

        source = np.arange(3., dtype=np.float64).view(CyclicArray)
        source_reference = weakref.ref(source)
        source.guard = DoubleBuffer(source)
        del source
        gc.collect()
        self.assertIsNone(source_reference())

    def test_shape_metadata_is_a_snapshot(self):
        source = np.arange(6., dtype=np.float64)
        guard = DoubleBuffer(source)
        source.shape = (2, 3)
        self.assertEqual(guard.shape, (6,))
        self.assertEqual(guard.strides, (8,))
        self.assertEqual(memoryview(guard).shape, (6,))
        self.assertEqual(guard.tolist(), [0., 1., 2., 3., 4., 5.])

    def test_unaligned_data_uses_safe_byte_access(self):
        source = np.ndarray(shape=(3,), buffer=bytearray(25), offset=1, dtype=np.float64)
        source[:] = [1.25, -2.5, 3.75]
        self.assertFalse(source.flags.aligned)
        guard = DoubleBuffer(source[::-1])
        self.assertEqual(guard.tolist(), [3.75, -2.5, 1.25])
        self.assertEqual(guard.copy().tolist(), [3.75, -2.5, 1.25])

    def test_empty_shapes_keep_metadata_and_export_safely(self):
        for shape in ((0,), (0, 3), (4, 0)):
            for copy in (False, True):
                with self.subTest(shape=shape, copy=copy):
                    guard = DoubleBuffer(np.empty(shape, dtype=np.float64), copy=copy)
                    self.assertEqual(guard.shape, shape)
                    self.assertEqual(guard.size, 0)
                    self.assertEqual(memoryview(guard).shape, shape)
                    self.assertEqual(memoryview(guard).nbytes, 0)
                    with self.assertRaises(IndexError):
                        guard[0] if len(shape) == 1 else guard[0, 0]

    def test_wrong_rank_or_dtype_is_rejected_and_releases_exports(self):
        for source in (np.ones((), dtype=np.float64), np.ones((1, 1, 1), dtype=np.float64),
                       np.ones(3, dtype=np.float32), np.ones(3, dtype='>f8')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                DoubleBuffer(source)
        source = array.array('i', [1, 2])
        with self.assertRaises(ValueError):
            DoubleBuffer(source)
        source.append(3)  # Constructor failure must release the acquired export.

    def test_overflowing_strides_are_rejected_before_reading(self):
        source = np.ones(1, dtype=np.float64)
        invalid = np.lib.stride_tricks.as_strided(
            source, shape=(3,), strides=(np.iinfo(np.intp).max,))
        with self.assertRaises(OverflowError):
            DoubleBuffer(invalid)
        invalid_matrix = np.lib.stride_tricks.as_strided(
            source, shape=(2, 2), strides=(np.iinfo(np.intp).max, np.iinfo(np.intp).max))
        with self.assertRaises(OverflowError):
            DoubleBuffer(invalid_matrix)

    def test_indices_and_reinitialization_are_checked(self):
        guard = DoubleBuffer(np.arange(6., dtype=np.float64).reshape(2, 3))
        self.assertEqual(guard[-1, -1], 5.)
        for index in ((2, 0), (0, 3), (-3, 0), (0,), 0, (1 << 100, 0)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                guard[index]
        with self.assertRaises(TypeError):
            guard[0., 1]
        with self.assertRaises(RuntimeError):
            guard.__init__(np.ones(1, dtype=np.float64))


if __name__ == '__main__':
    unittest.main()
