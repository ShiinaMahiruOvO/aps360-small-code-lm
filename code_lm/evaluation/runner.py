"""Execute grading jobs through an external isolated executor."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


class IsolatedGrader:
    def __init__(self, command):
        if not command or not command.strip():
            raise ValueError('An external isolated grader command is required')
        self.command = command

    def __call__(self, jobs):
        if not 1 <= len(jobs) <= 2048:
            raise ValueError('Expected 1–2048 grading jobs')
        with tempfile.TemporaryDirectory(prefix='bst-grader-') as temporary:
            root = Path(temporary)
            source, target = root/'input.json', root/'results.jsonl'
            source.write_text(json.dumps(jobs))
            # Submit to an isolated executor; never execute generated answers on the host.
            args = [arg.format(input=str(source), output=str(target))
                    for arg in shlex.split(self.command)]
            subprocess.run(args, check=True, timeout=4800)
            text = target.read_text()
            return (json.loads(text) if text.lstrip().startswith('[')
                    else [json.loads(line) for line in text.splitlines()])
