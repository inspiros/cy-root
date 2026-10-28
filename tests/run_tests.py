#!/usr/bin/env python3
"""Run unittest modules in bounded, independent subprocesses.

Use an installed editable package for source work. For wheel checks, invoke this
file by absolute path from outside the checkout and pass --installed-prefix.
Only the selected test directory is added for discovery, never its parent repo.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import unittest


THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
)
COUNT_FIELDS = (
    'tests_run', 'failures', 'errors', 'skipped', 'expected_failures',
    'unexpected_successes', 'discovery_errors',
)


def nonnegative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be an integer') from error
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def positive_integer(value: str) -> int:
    number = nonnegative_integer(value)
    if number == 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--jobs', type=positive_integer, default=min(4, os.cpu_count() or 1),
                        help='maximum concurrent module processes (default: min(4, CPU count))')
    result.add_argument('--start-directory', type=Path, default=Path(__file__).resolve().parent,
                        help='test discovery directory (default: directory containing this script)')
    result.add_argument('--pattern', default='test_*.py', help='test filename glob (default: test_*.py)')
    result.add_argument('--log-directory', type=Path, default=Path.cwd() / 'test-logs',
                        help='verbose module logs and summary.json (default: ./test-logs)')
    result.add_argument('--expected-tests', type=nonnegative_integer,
                        help='fail unless the aggregate tests-run count equals this value')
    result.add_argument('--require-no-skips', action='store_true', help='fail when any tests are skipped')
    result.add_argument('--installed-prefix', type=Path, nargs='?', const=Path(sys.prefix),
                        help='require installed imports within this prefix (flag alone uses the interpreter prefix)')
    result.add_argument('--self-check', action='store_true',
                        help='exercise this runner with isolated temporary test fixtures')
    result.add_argument('--_worker', help=argparse.SUPPRESS)
    result.add_argument('--_result-file', type=Path, help=argparse.SUPPRESS)
    return result


def discover_modules(start: Path, pattern: str) -> list[tuple[str, Path]]:
    """Mirror unittest's file/package traversal without importing test modules."""
    if not start.is_dir():
        raise ValueError(f'test directory does not exist: {start}')
    if not pattern or '/' in pattern or '\\' in pattern:
        raise ValueError('--pattern must be a nonempty filename glob')
    modules = []
    for directory, children, files in os.walk(start):
        directory = Path(directory)
        children[:] = sorted(child for child in children
                             if (directory / child / '__init__.py').is_file())
        for filename in sorted(files):
            if not filename.endswith('.py') or not fnmatch(filename, pattern):
                continue
            path = directory / filename
            parts = path.relative_to(start).with_suffix('').parts
            if not all(part.isidentifier() for part in parts):
                raise ValueError(f'test module must have an importable Python name: {path}')
            modules.append(('.'.join(parts), path))
    modules.sort()
    if not modules:
        raise ValueError(f'no test modules match {pattern!r} in {start}')
    return modules


def within(path: Path, prefix: Path) -> bool:
    return path.resolve().is_relative_to(prefix)


def check_installed_origin(prefix: Path) -> None:
    """Fail closed on source imports, including modules imported during tests."""
    package = importlib.import_module('cyroot')
    for name, module in sorted(tuple(sys.modules.items())):
        if name != 'cyroot' and not name.startswith('cyroot.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            raise RuntimeError(f'{name} has no verifiable installed module origin')
        if filename is not None and not within(Path(filename), prefix):
            raise RuntimeError(f'{name} loaded outside --installed-prefix {prefix}: {filename}')
        for entry in getattr(module, '__path__', ()):
            if not within(Path(entry), prefix):
                raise RuntimeError(f'{name} package path outside --installed-prefix {prefix}: {entry}')
    if not getattr(package, '__file__', None):
        raise RuntimeError('cyroot has no verifiable installed package origin')
    print(f'Installed provenance verified: {package.__file__}', flush=True)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_ids: list[str] = []

    def startTest(self, test):
        self.test_ids.append(test.id())
        super().startTest(test)


class ImportFailure(unittest.TestCase):
    """An arbitrary exception while importing a test module is a test error."""
    def __init__(self, module, error):
        super().__init__()
        self.module_name = module
        self.import_error = error

    def id(self):
        return f'unittest.loader._FailedTest.{self.module_name}'

    def runTest(self):
        raise self.import_error


def empty_report(module: str) -> dict:
    return dict(module=module, **dict.fromkeys(COUNT_FIELDS, 0), test_ids=[], successful=False)


def worker(args: argparse.Namespace) -> int:
    report = empty_report(args._worker)
    try:
        # unittest modules may import sibling test helpers. Add the discovery
        # directory only; the repository containing tests is never added.
        sys.path.insert(0, str(args.start_directory))
        if args.installed_prefix is not None:
            check_installed_origin(args.installed_prefix)
        loader = unittest.TestLoader()
        try:
            suite = loader.loadTestsFromName(args._worker)
        except BaseException as error:
            loader.errors.append(traceback.format_exc())
            suite = unittest.TestSuite((ImportFailure(args._worker, error),))
        result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
        report.update(
            tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
            unexpected_successes=len(result.unexpectedSuccesses), discovery_errors=len(loader.errors),
            test_ids=result.test_ids, successful=result.wasSuccessful() and not loader.errors,
        )
        if args.installed_prefix is not None:
            check_installed_origin(args.installed_prefix)
    except BaseException:
        report['worker_error'] = traceback.format_exc()
        traceback.print_exc()
    args._result_file.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0 if report['successful'] and 'worker_error' not in report else 1


def run_module(item: tuple[str, Path], args: argparse.Namespace, environment: dict) -> dict:
    name, path = item
    log_path = args.log_directory / f'{name}.log'
    result_path = args.log_directory / f'{name}.json'
    result_path.unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--_worker', name,
               '--start-directory', str(args.start_directory), '--_result-file', str(result_path)]
    if args.installed_prefix is not None:
        command += ['--installed-prefix', str(args.installed_prefix)]
    report = empty_report(name)
    try:
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if not result_path.is_file():
            raise RuntimeError(f'worker exited with code {process.returncode} without writing a result')
        report = json.loads(result_path.read_text(encoding='utf-8'))
        if report.get('module') != name or any(type(report.get(key)) is not int or report[key] < 0
                                             for key in COUNT_FIELDS):
            raise RuntimeError('worker wrote an invalid result report')
        if (type(report.get('successful')) is not bool or
                not isinstance(report.get('test_ids'), list) or
                not all(isinstance(value, str) for value in report['test_ids'])):
            raise RuntimeError('worker wrote invalid success/coverage fields')
        if process.returncode not in (0, 1) or (process.returncode == 0) != bool(report['successful']):
            if 'worker_error' not in report:
                report['worker_error'] = f'worker exit code {process.returncode} disagrees with its result'
        report['exit_code'] = process.returncode
    except Exception as error:
        report = empty_report(name)
        report['worker_error'] = str(error)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nRunner error: {error}\n')
    report.update(path=str(path), log=str(log_path))
    return report


def run(args: argparse.Namespace) -> int:
    try:
        modules = discover_modules(args.start_directory, args.pattern)
        args.log_directory.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as error:
        print(f'Discovery error: {error}', file=sys.stderr)
        return 2
    environment = os.environ.copy()
    for name in THREAD_VARIABLES:
        environment.setdefault(name, '1')
    jobs = min(args.jobs, len(modules))
    print(f'Running {len(modules)} modules with {jobs} process worker(s). Logs: {args.log_directory}', flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        reports = list(executor.map(lambda item: run_module(item, args, environment), modules))
    summary = {key: sum(report[key] for report in reports) for key in COUNT_FIELDS}
    summary.update(modules=len(reports), worker_errors=sum('worker_error' in report for report in reports),
                   reports=reports)
    issues = []
    if any(not report['successful'] or 'worker_error' in report for report in reports):
        issues.append('one or more test modules failed')
    if summary['tests_run'] == 0:
        issues.append('no tests ran')
    if args.expected_tests is not None and summary['tests_run'] != args.expected_tests:
        issues.append(f"expected {args.expected_tests} tests, ran {summary['tests_run']}")
    if args.require_no_skips and summary['skipped']:
        issues.append(f"--require-no-skips rejected {summary['skipped']} skipped test(s)")
    summary.update(successful=not issues, issues=issues)
    (args.log_directory / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for report in reports:
        status = 'OK' if report['successful'] and 'worker_error' not in report else 'FAIL'
        print(f"{status} {report['module']}: tests={report['tests_run']}, failures={report['failures']}, "
              f"errors={report['errors']}, skips={report['skipped']}; {report['log']}")
        if 'worker_error' in report:
            print(f"  Worker error: {report['worker_error'].strip()}")
    print(f"TOTAL: modules={summary['modules']}, tests={summary['tests_run']}, failures={summary['failures']}, "
          f"errors={summary['errors']}, skips={summary['skipped']}, expected_failures={summary['expected_failures']}, "
          f"unexpected_successes={summary['unexpected_successes']}, discovery_errors={summary['discovery_errors']}, "
          f"worker_errors={summary['worker_errors']}")
    for issue in issues:
        print(f'FAIL: {issue}', file=sys.stderr)
    return 1 if issues else 0


def self_check() -> int:
    """Verify subprocess isolation, aggregation and failure handling without cyroot."""
    with tempfile.TemporaryDirectory(prefix='cyroot-runner-check-') as temporary:
        root = Path(temporary)
        good = root / 'good'; good.mkdir()
        (good / 'test_one.py').write_text(
            'import unittest\nclass One(unittest.TestCase):\n'
            ' def test_first(self): self.assertTrue(True)\n'
            ' def test_second(self): self.assertEqual(2,2)\n', encoding='utf-8')
        (good / 'test_two.py').write_text(
            'import unittest\nclass Two(unittest.TestCase):\n'
            ' def test_pass(self): pass\n'
            ' @unittest.skip("fixture")\n def test_skip(self): pass\n'
            ' @unittest.expectedFailure\n def test_expected(self): self.fail("fixture")\n', encoding='utf-8')
        def invoke(label, directory, jobs=2, extra=(), expected=0, environment=None):
            logs = root / label
            command = [sys.executable, str(Path(__file__).resolve()), '--start-directory', str(directory),
                       '--jobs', str(jobs), '--log-directory', str(logs), *extra]
            result = subprocess.run(command, cwd=root, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode != expected:
                raise AssertionError(f'{label}: exit {result.returncode}, expected {expected}\n{result.stdout}')
            summary = json.loads((logs / 'summary.json').read_text()) if (logs / 'summary.json').is_file() else None
            return summary
        serial = invoke('serial', good, jobs=1, extra=('--expected-tests', '5'))
        parallel = invoke('parallel', good, jobs=2, extra=('--expected-tests', '5'))
        for key in (*COUNT_FIELDS, 'modules', 'worker_errors', 'successful'):
            assert serial[key] == parallel[key], key
        assert [report['test_ids'] for report in serial['reports']] == [report['test_ids'] for report in parallel['reports']]
        assert serial['tests_run'] == 5 and serial['skipped'] == 1 and serial['expected_failures'] == 1
        assert len({identifier for report in serial['reports'] for identifier in report['test_ids']}) == 5
        invoke('no-skips', good, extra=('--require-no-skips',), expected=1)
        invoke('wrong-count', good, extra=('--expected-tests', '6'), expected=1)
        bad = root / 'bad'; bad.mkdir()
        (bad / 'test_fail.py').write_text(
            'import unittest\nclass Bad(unittest.TestCase):\n'
            ' def test_fail(self): self.fail("fixture")\n'
            ' def test_error(self): raise RuntimeError("fixture")\n'
            ' @unittest.expectedFailure\n def test_unexpected(self): pass\n', encoding='utf-8')
        failure = invoke('failures', bad, expected=1)
        assert (failure['tests_run'], failure['failures'], failure['errors'], failure['unexpected_successes']) == (3, 1, 1, 1)
        broken = root / 'broken'; broken.mkdir()
        (broken / 'test_import.py').write_text('raise RuntimeError("fixture import failure")\n', encoding='utf-8')
        imported = invoke('import-error', broken, expected=1)
        assert imported['errors'] == 1 and imported['discovery_errors'] == 1
        crash = root / 'crash'; crash.mkdir()
        (crash / 'test_crash.py').write_text('import os\nos._exit(23)\n', encoding='utf-8')
        crashed = invoke('crash', crash, expected=1)
        assert crashed['worker_errors'] == 1 and crashed['tests_run'] == 0
        invoke('missing-directory', root / 'absent', expected=2)
        empty = root / 'empty'; empty.mkdir()
        invoke('empty-directory', empty, expected=2)
        prefix = root / 'installed'; package = prefix / 'cyroot'; package.mkdir(parents=True)
        (package / '__init__.py').write_text('# isolated provenance fixture\n', encoding='utf-8')
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(prefix)
        invoke('provenance-pass', good, extra=('--installed-prefix', str(prefix)), environment=environment)
        rejected = invoke('provenance-fail', good, extra=('--installed-prefix', str(root / 'elsewhere')),
                          environment=environment, expected=1)
        assert rejected['worker_errors'] == 2 and rejected['tests_run'] == 0
        rejected = invoke('provenance-current-prefix', good, extra=('--installed-prefix',),
                          environment=environment, expected=1)
        assert rejected['worker_errors'] == 2 and rejected['tests_run'] == 0
        late = root / 'late'; late.mkdir()
        (late / 'test_late.py').write_text(
            'import unittest,sys,types\nclass Late(unittest.TestCase):\n'
            ' def test_foreign_module(self):\n'
            '  module=types.ModuleType("cyroot.foreign")\n'
            f'  module.__file__={str(root / "foreign.py")!r}\n'
            '  sys.modules["cyroot.foreign"]=module\n', encoding='utf-8')
        rejected = invoke('provenance-late', late, extra=('--installed-prefix', str(prefix)),
                          environment=environment, expected=1)
        assert rejected['worker_errors'] == 1 and rejected['tests_run'] == 1
        print('Runner self-check passed: serial/parallel coverage, counts, skips, failures, imports, '
              'discovery, process crashes and installed provenance before/after tests.')
    return 0


def main() -> int:
    arguments = parser().parse_args()
    arguments.start_directory = arguments.start_directory.resolve()
    arguments.log_directory = arguments.log_directory.resolve()
    if arguments.installed_prefix is not None:
        arguments.installed_prefix = arguments.installed_prefix.resolve()
    if arguments._worker:
        if arguments._result_file is None:
            parser().error('--_worker requires --_result-file')
        return worker(arguments)
    return self_check() if arguments.self_check else run(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
