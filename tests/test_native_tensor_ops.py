"""Compile and exercise the standalone native Hessian contraction header."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cyroot


class TensorContractionHeaderTests(unittest.TestCase):
    def test_standalone_quadratic_forms_and_range_boundaries(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('A C++ compiler is required for the native header check.')
        source = Path(__file__).resolve().parent/'native'/'test_tensor_ops.cpp'
        include = Path(cyroot.__file__).resolve().parent/'native'
        with tempfile.TemporaryDirectory(prefix='cyroot-tensor-ops-') as name:
            executable = Path(name)/('check.exe' if os.name == 'nt' else 'check')
            command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                       '-fno-exceptions', '-fno-rtti', f'-I{include}', str(source),
                       '-o', str(executable)]
            environment = os.environ.copy()
            if environment.get('CYROOT_NATIVE_SANITIZERS') == '1':
                command.extend(['-g', '-O1', '-fno-omit-frame-pointer',
                                '-fsanitize=address,undefined'])
                environment.setdefault('ASAN_OPTIONS', 'detect_leaks=0:halt_on_error=1')
                environment.setdefault('UBSAN_OPTIONS', 'halt_on_error=1:print_stacktrace=1')
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True,
                           text=True, env=environment)


if __name__ == '__main__':
    unittest.main()
