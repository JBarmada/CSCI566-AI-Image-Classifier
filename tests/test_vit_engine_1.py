from vit_test_support import *


class NotebookTests(Fixture):
    def test_invalid_saved_protocol_is_not_adopted(self):
        m = self.m
        directory = self.root / 'invalid-protocol'
        directory.mkdir()
        m.atomic_json(directory / 'config.json', {'learning_rate': .001, 'protocol_hash': 'tampered'})
        with self.assertRaisesRegex(ValueError, 'Saved configuration hash is invalid'):
            m.resolve_run_config(directory, {'learning_rate': .001, 'protocol_hash': 'new'})

    def test_paths_hashes_and_leakage(self):
        m = self.m
        with self.assertRaises(ValueError):
            m.resolve_image_path('../escape.png')
        with self.assertRaises(ValueError):
            m.resolve_image_path('/tmp/elsewhere.png')
        csv = m.MANIFEST_ROOT / 'holdout_ADM_k2_rep0/val.csv'
        original = csv.read_bytes()
        try:
            val = pd.read_csv(csv)
            train = self.frames['train']
            val.loc[0, ['path','sha256']] = train.loc[0, ['path','sha256']].to_list()
            val.to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                m.load_experiment('holdout_ADM_k2_rep0')
        finally:
            csv.write_bytes(original)
        p = Path(self.frames['train'].iloc[0].resolved_path)
        before = p.read_bytes()
        try:
            p.write_bytes(before + b'changed')
            with self.assertRaisesRegex(ValueError, 'Content differs'):
                m.load_experiment('holdout_ADM_k2_rep0')
        finally:
            p.write_bytes(before)

    def test_cache_and_deterministic_degradations(self):
        m = self.m
        x, y = m.deterministic_features(self.frames['val'])
        calls = m.ENCODER.calls
        second, labels = m.deterministic_features(self.frames['val'])
        self.assertEqual(calls, m.ENCODER.calls)
        self.assertTrue(torch.equal(x, second))
        self.assertTrue(torch.equal(y, labels))
        row = self.frames['val'].iloc[0]
        for condition in ['clean', 'blur','jpeg','resize','noise','lowpass']:
            with Image.open(row.resolved_path) as im:
                transform = m.ImageTransform(m.PREPROCESS, condition=condition)
                first = transform(im.convert('RGB'), row.sha256)
                second = transform(im.convert('RGB'), row.sha256)
            self.assertTrue(torch.equal(first,second), condition)
            self.assertTrue(torch.isfinite(first).all())
        a = m.feature_identity(self.frames['val'], 'clean')
        b = m.feature_identity(self.frames['val'].iloc[::-1], 'clean')
        self.assertNotEqual(m.digest_json(a), m.digest_json(b))

    def test_training_resume_evaluation_and_exports(self):
        m = self.m
        for mode in ['baseline','augmented']:
            config = self.make_config(mode)
            config['preprocessing'] = 'Compose(<function _convert_image_to_rgb at 0x111abc>)'
            config['protocol_hash'] = m.raw_digest_json({k:v for k,v in config.items() if k != 'protocol_hash'})
            full = self.root / f'full-{mode}'
            resumed = self.root / f'resumed-{mode}'
            full.mkdir(); resumed.mkdir()
            m.atomic_json(resumed / 'config.json', config)
            m.train_one(full, self.frames, config)
            original_save = m.atomic_torch
            def interrupt_after_epoch(path, value):
                original_save(path, value)
                if Path(path).name == 'last.pt' and value['state']['epoch'] == 1:
                    raise KeyboardInterrupt('synthetic interruption')
            m.atomic_torch = interrupt_after_epoch
            try:
                with self.assertRaises(KeyboardInterrupt):
                    m.train_one(resumed, self.frames, config)
            finally:
                m.atomic_torch = original_save
            restarted = {**config, 'preprocessing': config['preprocessing'].replace('0x111abc', '0x222def')}
            restarted['protocol_hash'] = m.digest_json({k:v for k,v in restarted.items() if k != 'protocol_hash'})
            restarted = m.resolve_run_config(resumed, restarted)
            self.assertEqual(restarted['protocol_hash'], config['protocol_hash'])
            m.train_one(resumed, self.frames, restarted)
            a = torch.load(full/'best_head.pt', weights_only=True)
            b = torch.load(resumed/'best_head.pt', weights_only=True)
            for key in a['head_state_dict']:
                self.assertTrue(torch.equal(a['head_state_dict'][key], b['head_state_dict'][key]), (mode,key))
            self.assertEqual(len(pd.read_csv(resumed/'training_history.csv')), 2)
            for condition in ['clean','noise']:
                m.evaluate_one(full, self.frames, config, condition)
                calls = m.ENCODER.calls
                m.evaluate_one(full, self.frames, config, condition)
                self.assertEqual(calls, m.ENCODER.calls)
                _, _, directory = m.evaluation_identity(full, config, condition)
                pred = pd.read_csv(directory/'test_predictions.csv')
                self.assertEqual(len(pred), 4)
                self.assertTrue({'image_id','label','prob_fake','source_generator','experiment','mode','seed'} <= set(pred))
                self.assertTrue((directory/'test_diagnostics.png').is_file())
                if condition == 'noise':
                    (directory/'test_predictions.csv').unlink()
                    with self.assertRaises(FileNotFoundError):
                        m.evaluate_one(full, self.frames, config, condition)
