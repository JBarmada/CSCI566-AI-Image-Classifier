from vit_test_support import *


class NotebookTests(Fixture):
    def test_exclusive_lock(self):
        m = self.m
        d = self.root/'lock-test'
        d.mkdir()
        with m.run_lock(d):
            with self.assertRaises(RuntimeError):
                with m.run_lock(d):
                    pass
        self.assertFalse((d/'RUNNING.lock').exists())

    def test_plan_runner_resume_limit_and_summaries(self):
        m = self.m
        # Exercise the actual plan, runner, and aggregation cells on tiny images.
        m.OUTPUT_ROOT = self.root / 'plan-results'
        m.HOLDOUTS = ['ADM']; m.K_VALUES = [1, 2]; m.REPEATS = [0]
        m.TRAINING_MODES = ['baseline', 'augmented']
        m.EVAL_CONDITIONS = ['clean']
        m.MAX_RUNS_THIS_SESSION = 2
        m.package_versions = lambda: {'test': 'synthetic CPU'}
        code = [''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code']
        plan = next(c for c in code if c.startswith('validate_settings()'))
        runner = next(c for c in code if c.startswith('session_rows ='))
        summary = next(c for c in code if c.startswith('rows, missing ='))
        exec(compile(plan, 'plan-cell', 'exec'), m.__dict__)
        self.assertEqual(len(m.PLAN), 4)
        exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        self.assertEqual(len(list(m.OUTPUT_ROOT.glob('runs/*/TRAINING_COMPLETE.json'))), 2)
        exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        self.assertEqual(len(list(m.OUTPUT_ROOT.glob('runs/*/TRAINING_COMPLETE.json'))), 4)
        calls = m.ENCODER.calls
        exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        self.assertEqual(m.ENCODER.calls, calls)
        exec(compile(summary, 'summary-cell', 'exec'), m.__dict__)
        results = pd.read_csv(m.SUMMARY_DIR/'all_results.csv')
        paired = pd.read_csv(m.SUMMARY_DIR/'augmentation_deltas.csv')
        self.assertEqual(len(results), 4)
        self.assertEqual(len(paired), 2)
        self.assertTrue((m.SUMMARY_DIR/'baseline_clean_auroc_heatmap.png').is_file())
        # Changing training settings must reject existing run identities.
        m.LEARNING_RATE *= 2
        with self.assertRaisesRegex(ValueError, 'Protocol changed'):
            exec(compile(plan, 'plan-cell', 'exec'), m.__dict__)
        m.LEARNING_RATE /= 2
