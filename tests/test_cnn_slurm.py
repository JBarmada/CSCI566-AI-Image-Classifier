"""Static checks for the CNN Slurm workflow."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
CNN = ROOT / "code/cnn"


class CnnSlurmTests(unittest.TestCase):
    def test_array_partition_and_throttle(self):
        source = (CNN / "run_benchmark.slurm").read_text()
        self.assertIn("#SBATCH --array=0-7%2", source)
        self.assertIn("${holdouts[$SLURM_ARRAY_TASK_ID]}", source)
        for holdout in ["ADM", "BigGAN", "glide", "Midjourney", "VQDM", "wukong"]:
            self.assertIn(holdout, source)

    def test_training_job_has_gpu_and_signal(self):
        source = (CNN / "run_cnn_master.slurm").read_text()
        self.assertIn("#SBATCH --gpus-per-task=1", source)
        self.assertIn("#SBATCH --signal=B:USR1@120", source)
        self.assertIn("kill -USR1", source)

    def test_finalizer_is_cpu_and_dependency_is_documented(self):
        source = (CNN / "finalize_benchmark.slurm").read_text()
        self.assertIn("--dependency=afterok:ARRAY_JOB_ID", source)
        self.assertNotIn("--gpus-per-task", source)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_shell_syntax(self):
        for name in [
            "run_benchmark.slurm", "run_cnn_master.slurm", "finalize_benchmark.slurm"
        ]:
            with self.subTest(name=name):
                result = subprocess.run(
                    ["bash", "-n", str(CNN / name)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
