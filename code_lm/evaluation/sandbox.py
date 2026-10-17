"""Trusted runner intended ONLY for a credential-free, network-blocked Linux sandbox."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET

CHILD = '''import resource, runpy, sys
# Distinct limits make the CPU deadline observable as SIGXCPU, not ambiguous SIGKILL.
resource.setrlimit(resource.RLIMIT_CPU, (4, 5))
resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
sys.argv = ['pytest', '-q', '-x', '-p', 'no:cacheprovider', '--disable-warnings', '--junitxml=result.xml', 'test_solution.py']
runpy.run_module('pytest', run_name='__main__')
'''


def collect_result(root, code):
    """Read a finished child's report; distinguish test failures from runner failures."""
    log = (root / 'process.log').read_text(errors='replace')[-1500:]
    if code == -getattr(signal, 'SIGXCPU', 24):
        return {'status': 'timeout', 'reason': 'cpu_limit', 'exit_code': code, 'log_tail': log}
    try:
        suites = ET.parse(root / 'result.xml').getroot()
        cases = list(suites.iter('testcase'))
        failures = sum(bool(list(c.iter('failure'))) for c in cases)
        errors = sum(bool(list(c.iter('error'))) for c in cases)
        skipped = sum(bool(list(c.iter('skipped'))) for c in cases)
        # A signal or pytest internal error is not evidence of ordinary test failure,
        # even if a partial XML report happens to exist.
        if code not in (0, 1, 2):
            return {'status': 'runner_error', 'exit_code': code, 'log_tail': log}
        return {'status': 'pass' if code == 0 and cases and not (failures or errors or skipped) else 'fail',
                'exit_code': code, 'tests': len(cases), 'failures': failures, 'errors': errors,
                'skipped': skipped, 'log_tail': log if code else ''}
    except (FileNotFoundError, ET.ParseError):
        return {'status': 'runner_error', 'exit_code': code, 'log_tail': log}


def execute(row, index):
    uid = 10000 + index
    root = Path('/tmp') / f'case-{index}'
    root.mkdir(mode=0o700)
    code = row['answer']
    for name, text in [('solution.py', code), ('test_solution.py', row['tests'])]:
        path = root / name
        path.write_text(text)
        os.chown(path, uid, uid)
    os.chown(root, uid, uid)
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(root),
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'}
    with (root / 'process.log').open('wb') as log:
        proc = subprocess.Popen([sys.executable, '-I', '-c', CHILD], cwd=root, env=env,
                                user=uid, group=uid, extra_groups=[], start_new_session=True,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return {'status': 'timeout', 'reason': 'wall_limit'}
    return collect_result(root, code)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def main():
    jobs = json.loads(Path('/tmp/input.json').read_text())
    if not 1 <= len(jobs) <= 2048 or len({r['id'] for r in jobs}) != len(jobs):
        raise ValueError('Expected at most 2048 unique jobs')
    for row in jobs:
        if fingerprint({k: v for k, v in row.items() if k != 'job_sha256'}) != row['job_sha256']:
            raise ValueError('Corrupted job')
    with Path('/tmp/results.jsonl').open('x') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(execute, row, index): row for index, row in enumerate(jobs)}
        for count, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'status': 'runner_error', 'error': str(exc)}
            stream.write(json.dumps({'id': row['id'], 'job_sha256': row['job_sha256'], 'result': result}) + '\n')
            stream.flush()
            if count % 25 == 0 or count == len(jobs):
                print(json.dumps({'finished': count, 'total': len(jobs)}), flush=True)


if __name__ == '__main__':
    if sys.platform != 'linux' or os.getuid() != 0 or os.environ.get('BST_ISOLATED_GRADER') != '1':
        raise SystemExit('Use the isolated Linux grader, never the development host')
    main()
