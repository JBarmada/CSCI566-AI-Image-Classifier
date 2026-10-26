"""Exercise the actual Slurm launcher with fake modules, Conda, and Python."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'code/vit/run_vit_master.slurm'
if os.name == 'nt':
    git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
    BASH = str(git_bash) if git_bash.is_file() else None
else:
    BASH = shutil.which('bash')

WRAPPER = r'''
export SLURM_SUBMIT_DIR="$1"
export FAKE_CONDA_BASE="$2"
export LAUNCH_RECORD="$3"
export RUNNER_EXIT_CODE="$4"
export SLURM_JOB_ID=1234
export SLURM_CPUS_PER_TASK=4
module() { :; }
conda() { if [[ "$1" == info ]]; then printf '%s\n' "$FAKE_CONDA_BASE"; fi; }
python() {
    {
        if command -v cygpath >/dev/null 2>&1; then cygpath -m "$PWD";
        else printf '%s\n' "$PWD"; fi
        printf '%s\n' "$@"
    } > "$LAUNCH_RECORD"
    return "$RUNNER_EXIT_CODE"
}
source "$5" --preflight-only
'''


@unittest.skipUnless(BASH, 'Bash is required for Slurm shell checks')
class SlurmTests(unittest.TestCase):
    def run_launcher(self, location, exit_code=0):
        with tempfile.TemporaryDirectory() as td:
            temporary = Path(td)
            repo = temporary / 'repo with spaces'
            (repo / 'code/vit').mkdir(parents=True)
            (repo / 'ViT_Master_Experiments.ipynb').write_text('{}')
            (repo / 'code/vit/run_vit_master.py').touch()
            # Keep the executing script outside the repo, as Slurm does.
            spool = temporary / 'slurm spool/job.sh'
            spool.parent.mkdir()
            spool.write_bytes(SCRIPT.read_bytes())
            conda = temporary / 'conda/etc/profile.d/conda.sh'
            conda.parent.mkdir(parents=True)
            conda.touch()
            submit = temporary / 'unrelated' if location == 'outside' else repo / location
            submit.mkdir(parents=True, exist_ok=True)
            record = temporary / 'launch.txt'
            result = subprocess.run(
                [BASH, '--noprofile', '--norc', '-c', WRAPPER, 'slurm-check',
                 submit.as_posix(), (temporary / 'conda').as_posix(), record.as_posix(),
                 str(exit_code), spool.as_posix()],
                capture_output=True, text=True, timeout=30,
            )
            if location == 'outside':
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Cannot find the ViT repository', result.stderr)
                self.assertFalse(record.exists())
                return
            self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
            lines = record.read_text().splitlines()
            self.assertEqual(Path(lines[0]).resolve(), repo.resolve())
            self.assertEqual(lines[1:], ['-u', 'code/vit/run_vit_master.py',
                                         '--recover-stale-locks', '--preflight-only'])

    def test_repository_root(self):
        self.run_launcher('.')

    def test_subdirectories(self):
        for location in ['code/vit', 'code/vit/deeper']:
            with self.subTest(location=location):
                self.run_launcher(location)

    def test_outside_checkout_has_clear_error(self):
        self.run_launcher('outside')

    def test_runner_exit_code_is_preserved(self):
        self.run_launcher('code/vit', exit_code=7)


if __name__ == '__main__':
    unittest.main()
