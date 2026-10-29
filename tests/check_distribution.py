"""Inspect built release contents without importing the package."""

import sys
import re
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path


def check_distribution(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
            text = archive.read(metadata).decode()
            assert "Requires-Python: >=3.11,<3.15" in text or "Requires-Python: <3.15,>=3.11" in text
        package_root = ""
    else:
        with tarfile.open(path) as archive:
            names = set(archive.getnames())
            metadata = next(name for name in names if name.count('/') == 1 and name.endswith('/PKG-INFO'))
            text = archive.extractfile(metadata).read().decode()
        package_root = next(iter(names)).split("/")[0] + "/"
    message = Parser().parsestr(text)
    readme = message.get_payload()
    assert message['Description-Content-Type'] == 'text/markdown'
    asset_base = f"https://raw.githubusercontent.com/inspiros/cy-root/v{message['Version']}/assets/"
    for image in ('cyroot_logo.svg', 'hero.svg', 'scalar_steps.gif', 'vector_steps.gif', 'vector_surface.gif'):
        assert f'src="{asset_base}{image}"' in readme, image
    assert f'srcset="{asset_base}cyroot_logo-dark.svg"' in readme
    assert 'media="(prefers-color-scheme: dark)"' in readme
    assert not re.search(r'<(?:img|source)\b[^>]*\s(?:src|srcset)=[\"\'](?:\./)?assets/', readme)
    assert '/master/assets/' not in readme
    relative = [target for target in re.findall(r'\]\(\s*<?([^\s)>]+)', readme)
                if not target.startswith(('http://', 'https://', '#'))]
    assert not relative, relative
    assert f'https://github.com/inspiros/cy-root/blob/v{message["Version"]}/LICENSE' in readme
    assert not any('sympy' in line.lower() for line in text.splitlines()
                   if line.startswith('Requires-Dist:'))
    for name in ('py.typed', 'results.py', '__init__.pyi', 'return_types.pyi',
                 'scalar_bracketing.pyi', 'scalar_quasi_newton.pyi', 'scalar_newton.pyi',
                 'vector_bracketing.pyi', 'vector_quasi_newton.pyi', 'vector_newton.pyi',
                 'scalar_derivative_approximation.pyi', 'vector_derivative_approximation.pyi',
                 'native/broyden.pyi', 'native/vector.pyi'):
        assert package_root + 'cyroot/' + name in names
    for name in ('scalar', 'bracketing', 'secant', 'interpolation', 'homeier', 'hybisect',
                 'higher_order', 'householder_solver', 'scalar_derivatives',
                 'higher_derivatives', 'higher_mixed', 'derivatives', 'hessians',
                 'tensor_callbacks', 'rectangular', 'least_squares', 'vrahatis',
                 'vector_chebyshev', 'objective', 'objective_derivatives',
                 'levenberg_marquardt', 'bfgs', '_buffers', 'complex_callbacks'):
        assert package_root + 'cyroot/native/' + name + '.pyi' in names, name
    for name in ("higher_mixed.pyx", "higher_mixed.pxd", "higher_mixed_ops.pxd", "higher_mixed.hpp"):
        assert package_root + "cyroot/native/" + name in names
    for name in ('differentiation.pyx', 'differentiation.pxd', 'differentiation.pyi', 'differentiation.hpp'):
        assert package_root + 'cyroot/native/' + name in names
    for name in ("levenberg_marquardt.pyx", "levenberg_marquardt.pxd", "lm_math.hpp", "lm_math.pxd"):
        assert package_root + "cyroot/native/" + name in names
    for name in ("bfgs.pyx", "bfgs.pxd", "bfgs_math.hpp", "bfgs_math.pxd"):
        assert package_root + "cyroot/native/" + name in names
    for name in ("lbfgs_math.hpp", "lbfgs_math.pxd", "dogleg_math.hpp", "dogleg_math.pxd"):
        assert package_root + "cyroot/native/" + name in names
    for name in ("damped_newton.pyx", "damped_newton.pxd", "damped_newton.pyi", "damped_newton.hpp"):
        assert package_root + "cyroot/native/" + name in names
    for name in ("jvp.pyx", "jvp.pxd", "jvp.pyi", "jvp.hpp", "jvp_callbacks.h"):
        assert package_root + "cyroot/native/" + name in names
    for name in ("newton_krylov.pyx", "newton_krylov.pxd", "newton_krylov.pyi", "gmres.hpp", "lsqr.hpp"):
        assert package_root + "cyroot/native/" + name in names
    for name in ("vjp.pyx", "vjp.pxd", "vjp.pyi", "vjp_callbacks.h"):
        assert package_root + "cyroot/native/" + name in names
    assert package_root + "cyroot/fptr.pxd" in names
    assert package_root + "cyroot/_stopping.pxd" in names
    assert package_root + "cyroot/status.py" in names
    assert package_root + "cyroot/native/broyden.pxd" in names
    assert package_root + "cyroot/native/broyden_ops.pxd" in names
    assert package_root + "cyroot/native/broyden.hpp" in names
    assert package_root + "cyroot/native/barnes.hpp" in names
    assert package_root + "cyroot/native/barnes_ops.pxd" in names
    assert package_root + "cyroot/native/wolfe_bittner.hpp" in names
    assert package_root + "cyroot/native/wolfe_bittner_ops.pxd" in names
    assert package_root + "cyroot/native/robinson.hpp" in names
    assert package_root + "cyroot/native/robinson_ops.pxd" in names
    assert package_root + "cyroot/native/scalar.pxd" in names
    assert package_root + "cyroot/native/vector.pxd" in names
    assert package_root + "cyroot/native/linalg.pxd" in names
    assert package_root + "cyroot/native/least_squares.pxd" in names
    assert package_root + "cyroot/native/rectangular.pxd" in names
    assert package_root + "cyroot/native/derivatives.pxd" in names
    assert package_root + "cyroot/native/scalar_derivatives.pxd" in names
    assert package_root + "cyroot/native/secant.pxd" in names
    assert package_root + "cyroot/native/higher_order.pxd" in names
    assert package_root + "cyroot/native/higher_derivatives.pxd" in names
    assert package_root + "cyroot/native/householder.pxd" in names
    assert package_root + "cyroot/native/householder.hpp" in names
    assert package_root + "cyroot/native/householder_solver.pxd" in names
    assert package_root + "cyroot/native/hessians.pxd" in names
    assert package_root + "cyroot/native/tensor_callbacks.pxd" in names
    assert package_root + "cyroot/native/vector_chebyshev.pxd" in names
    assert package_root + "cyroot/native/tensor_ops.pxd" in names
    assert package_root + "cyroot/native/tensor_ops.hpp" in names
    assert package_root + "cyroot/native/callbacks.h" in names
    for module in ("vrahatis", "polygons"):
        for extension in (".pyx", ".pxd"):
            assert package_root + "cyroot/native/" + module + extension in names
    for module in ("bracketing", "interpolation", "complex_callbacks", "homeier", "hybisect"):
        for extension in (".pyx", ".pxd"):
            assert package_root + "cyroot/native/" + module + extension in names
    for header in ("interval_callbacks.h", "complex_callbacks.h"):
        assert package_root + "cyroot/native/" + header in names
    assert package_root + "cyroot/native/buffers.hpp" in names
    assert package_root + "cyroot/native/buffers.pxd" in names
    assert package_root + "cyroot/native/_buffers.pxd" in names
    assert package_root + "cyroot/ops/scalar_types.pxd" in names
    assert package_root + "cyroot/defaults.hpp" in names
    assert any(name.endswith("/LICENSE") or name == "LICENSE" for name in names)
    assert not any(".maintainer/" in name or "/develop/" in name or "/.venv/" in name for name in names)
    assert not any(name.endswith((".c", ".cpp"))
                   and not name.startswith(package_root + "tests/native/") for name in names)
    if path.suffix != ".whl":
        for name in ('tests/test_differentiation.py', 'tests/differentiation_consumer.pyx',
                     'examples/numerical_derivatives.py'):
            assert package_root + name in names
        for name in ('assets/cyroot_logo.svg', 'assets/cyroot_logo-dark.svg',
                     'assets/hero.svg', 'examples/generate_hero.py',
                     'examples/plotting.py', 'examples/scalar_steps.py', 'examples/vector_steps.py',
                     'assets/scalar_steps.gif', 'assets/vector_steps.gif',
                     'assets/vector_surface.gif'):
            assert package_root + name in names
        assert package_root + 'assets/logo.png' not in names
        assert not any('/resources/' in name for name in names)
        for name in ('tests/test_public_results.py', 'tests/typing/public_api_valid.py',
                     'tests/typing/public_api_invalid.py'):
            assert package_root + name in names
        for name in ("tests/test_optimization_interface.py", "tests/optimization_interface_consumer.pyx"):
            assert package_root + name in names
        for name in ("tests/test_native_optimization_extensions.py", "tests/native_optimization_extensions_consumer.pyx"):
            assert package_root + name in names
        for name in ("tests/test_native_bfgs.py", "tests/native_bfgs_consumer.pyx"):
            assert package_root + name in names
        for name in ("tests/test_native_lm.py", "tests/native_lm_consumer.pyx",
                     "docs/native-levenberg-marquardt.md", "examples/levenberg_marquardt.py"):
            assert package_root + name in names
        for name in ("docs/native-vrahatis.md", "tests/test_native_vrahatis.py",
                     "tests/native_vrahatis_consumer.pyx", "examples/native_vrahatis.pyx",
                     "examples/native_vrahatis_setup.py", "examples/native_vrahatis.md"):
            assert package_root + name in names
        for name in ("docs/scalar-native-migration.md", "tests/test_native_bracketing.py",
                     "tests/test_native_interpolation.py", "tests/native_interpolation_consumer.pyx",
                     "tests/test_native_hybisect.py", "tests/native_hybisect_consumer.pyx",
                     "tests/test_scalar_native_migration.py"):
            assert package_root + name in names
        for name in ("examples/pure_python_callback.py", "examples/pure_python_callback_setup.py",
                     "examples/pure_python_callback.md", "docs/compiling-callbacks.md"):
            assert package_root + name in names
        for name in ("tests/test_vector_difference_migration.py", "tests/vector_difference_consumer.pyx",
                     "docs/vector-derivative-migration.md"):
            assert package_root + name in names
        for name in ("tests/test_native_higher_mixed.py", "tests/native_higher_mixed_consumer.pyx",
                     "docs/native-higher-mixed.md", "examples/native_mixed_partials.py"):
            assert package_root + name in names
        for name in ("tests/test_rectangular_modified.py", "docs/native-rectangular-modified.md"):
            assert package_root + name in names
        for name in ("tests/test_rectangular_halley.py", "docs/native-rectangular-halley.md"):
            assert package_root + name in names
        for name in ("tests/test_rectangular_chebyshev.py", "docs/native-rectangular-chebyshev.md"):
            assert package_root + name in names
        for name in ("tests/test_rectangular_tensors.py", "docs/native-rectangular-tensors.md"):
            assert package_root + name in names
        for name in ("tests/test_rectangular_hessian.py", "docs/native-rectangular-hessian.md"):
            assert package_root + name in names
        for name in ("tests/test_native_wolfe_bittner.py", "tests/native/test_wolfe_bittner.cpp",
                     "docs/native-wolfe-bittner.md"):
            assert package_root + name in names
        for name in ('tests/test_native_robinson.py', 'tests/native/test_robinson.cpp',
                     'docs/native-robinson.md'):
            assert package_root + name in names
        for name in ('tests/test_native_barnes.py', 'tests/native/test_barnes.cpp',
                     'docs/native-barnes.md'):
            assert package_root + name in names
        for name in ('tests/test_native_steffensen.py', 'docs/native-steffensen.md'):
            assert package_root + name in names
        for name in ('tests/test_native_klement.py', 'tests/native/test_klement.cpp',
                     'docs/native-klement.md'):
            assert package_root + name in names
        for name in ('tests/test_vjp.py', 'tests/vjp_consumer.pyx', 'examples/rectangular_krylov.py'):
            assert package_root + name in names
        assert package_root + "tests/test_rectangular_newton.py" in names
        assert package_root + "tests/test_rectangular_jacobian.py" in names
        assert package_root + "tests/rectangular_jacobian_consumer.pyx" in names
        for name in ("tests/test_rectangular_callbacks.py", "tests/rectangular_callbacks_consumer.pyx",
                     "docs/native-rectangular-callbacks.md"):
            assert package_root + name in names
        for name in ("tests/test_native_least_squares.py", "tests/native_least_squares_consumer.pyx",
                     "docs/native-least-squares.md", "examples/native_least_squares.pyx",
                     "examples/native_least_squares_setup.py", "examples/native_least_squares.md"):
            assert package_root + name in names
        assert package_root + "cyroot/scalar_bracketing.pyx" in names
        assert package_root + "tests/test_matrix_ops.py" in names
        assert package_root + "tests/native/test_buffers.cpp" in names
        assert package_root + "examples/native_scalar.pyx" in names
        assert package_root + "examples/native_vector.pyx" in names
        assert package_root + "examples/native_jacobian.pyx" in names
        assert package_root + "examples/native_scalar_derivative.pyx" in names
        assert package_root + "examples/native_secant.pyx" in names
        assert package_root + "examples/native_higher_order.pyx" in names
        assert package_root + "docs/native-second-derivative.md" in names
        assert package_root + "tests/test_native_second_derivative.py" in names
        assert package_root + "tests/native_second_derivative_consumer.pyx" in names
        assert package_root + "tests/test_native_halley_family.py" in names
        assert package_root + "tests/native_halley_family_consumer.pyx" in names
        assert package_root + "tests/test_native_higher_derivative.py" in names
        assert package_root + "tests/native_higher_derivative_consumer.pyx" in names
        assert package_root + "tests/test_native_householder_recurrence.py" in names
        assert package_root + "tests/native_householder_recurrence_consumer.pyx" in names
        assert package_root + "docs/native-higher-derivative.md" in names
        assert package_root + "examples/native_higher_derivative.pyx" in names
        assert package_root + "tests/test_native_householder.py" in names
        assert package_root + "tests/test_householder_migration.py" in names
        assert package_root + "examples/native_householder.pyx" in names
        assert package_root + "docs/native-householder.md" in names
        assert package_root + "tests/test_native_hessian.py" in names
        assert package_root + "tests/native_hessian_consumer.pyx" in names
        assert package_root + "examples/native_hessian.pyx" in names
        assert package_root + "examples/native_hessian_setup.py" in names
        assert package_root + "docs/native-hessian.md" in names
        assert package_root + "tests/test_native_vector_chebyshev.py" in names
        assert package_root + "tests/native_vector_chebyshev_consumer.pyx" in names
        assert package_root + "tests/test_native_factor_overflow.py" in names
        assert package_root + "tests/test_native_tensor_ops.py" in names
        assert package_root + "tests/native/test_tensor_ops.cpp" in names
        assert package_root + "examples/native_vector_chebyshev.pyx" in names
        assert package_root + "examples/native_vector_chebyshev_setup.py" in names
        assert package_root + "docs/native-vector-chebyshev.md" in names
        assert package_root + "tests/test_native_vector_halley.py" in names
        assert package_root + "tests/native_vector_halley_consumer.pyx" in names
        assert package_root + "examples/native_vector_halley.pyx" in names
        assert package_root + "examples/native_vector_halley_setup.py" in names
        assert package_root + "docs/native-vector-halley.md" in names
        assert package_root + "tests/test_native_broyden_initialization.py" in names
        assert package_root + "tests/run_tests.py" in names
        assert package_root + "tests/test_native_broyden.py" in names
        assert package_root + "tests/native_broyden_consumer.pyx" in names
        assert package_root + "tests/native/test_broyden.cpp" in names
        assert package_root + "examples/native_broyden.pyx" in names
        assert package_root + "examples/native_broyden_setup.py" in names
        assert package_root + "docs/native-broyden.md" in names
        assert package_root + "tests/test_vector_api_migration.py" in names
        assert package_root + "docs/vector-higher-order-migration.md" in names
        assert package_root + "examples/vector_higher_order.py" in names
        assert package_root + "tests/test_native_vector_modified.py" in names
        assert package_root + "tests/test_native_vector_modified_range.py" in names
        assert package_root + "tests/native_vector_modified_consumer.pyx" in names
        assert package_root + "examples/native_vector_modified.pyx" in names
        assert package_root + "examples/native_vector_modified_setup.py" in names
        assert package_root + "docs/native-vector-modified.md" in names
    print(f"Checked {path.name}")


if __name__ == "__main__":
    distributions = [p for p in Path(sys.argv[1]).iterdir() if p.name.endswith((".whl", ".tar.gz"))]
    if not distributions:
        raise SystemExit("No distributions found")
    for distribution in distributions:
        check_distribution(distribution)
