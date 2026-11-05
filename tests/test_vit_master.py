from vit_test_support import *


class NotebookTests(Fixture):
    def test_default_plan_and_syntax(self):
        self.assertEqual(self.default_plan_size, 192)
        ids = [c['id'] for c in self.nb['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for i, c in enumerate(self.nb['cells']):
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), f'cell-{i}', 'exec')
        for cell in json.loads(NOTEBOOK.read_text())['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'notebook-wrapper', 'exec')

    def test_floating_point_auroc_tie_prefers_lower_loss(self):
        self.assertTrue(self.m.better_checkpoint(.9995663265306122, .615,
                                                 .9995663265306123, .638))
        self.assertFalse(self.m.better_checkpoint(.89, .1, .90, .5))
        self.assertTrue(self.m.better_checkpoint(.91, .9, .90, .5))

    def test_metrics_and_calibration_boundaries(self):
        m = self.m
        perfect = m.classification_metrics([0, 1, 0, 1], [0., 1., 0., 1.])
        self.assertEqual(perfect['auroc'], 1.)
        self.assertEqual(perfect['accuracy'], 1.)
        self.assertEqual(perfect['brier'], 0.)
        self.assertEqual(perfect['ece'], 0.)
        self.assertEqual(m.reliability_bins([0,1], [0.,1.])['count'].sum(), 2)
        self.assertEqual(m.classification_metrics([0,1], [.5,.5])['f1_fake'], 2/3)
        for bad in [[float('nan'), .2], [-.1, 1], [0, 1.1]]:
            with self.assertRaises(ValueError):
                m.classification_metrics([0,1], bad)

    def test_fingerprint_without_source_or_class_file(self):
        m = self.m
        original = m.ManifestImages
        class_source = None
        for cell in self.nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            for node in ast.parse(source).body:
                if isinstance(node, ast.ClassDef) and node.name == 'ManifestImages':
                    class_source = ast.get_source_segment(source, node)
        namespace = {**m.__dict__, '__name__': '__main__'}
        # Match notebook-style classes whose module has no usable source file.
        exec(compile('\n\n' + class_source, '<ipython-input-999>', 'exec'), namespace)
        try:
            m.ManifestImages = namespace['ManifestImages']
            with patch.object(m.inspect, 'getsource', side_effect=OSError('source code not available')):
                before = m.code_fingerprint()
                # Compare equivalent cell compilations, not file-vs-cell compiler
                # optimizations (Python 3.12 may optimize imported module attributes).
                exec(compile('\n\n\n' + class_source, '<ipython-input-1000>', 'exec'), namespace)
                m.ManifestImages = namespace['ManifestImages']
                self.assertEqual(before, m.code_fingerprint())
                def changed_length(self):
                    return 999
                m.ManifestImages.__len__ = changed_length
                self.assertNotEqual(before, m.code_fingerprint())
        finally:
            m.ManifestImages = original

    def test_restart_preprocessing_address_keeps_saved_checkpoint_identity(self):
        m = self.m
        directory = self.root / 'restart-address'
        directory.mkdir()
        payload = {'experiment': 'holdout_ADM_k1_rep0', 'mode': 'baseline',
                   'preprocessing': 'Compose(<function _convert_image_to_rgb at 0x7f123abc>)',
                   'learning_rate': .001, 'engine_hash': 'same-training-code',
                   'expected_counts': {0: 1400, 1: 1400}, 'environment': {'torch': 'unchanged'}}
        legacy_hash = m.raw_digest_json(payload)
        m.atomic_json(directory / 'config.json', {**payload, 'protocol_hash': legacy_hash,
                      'created_utc': 'original', 'gpu': 'P100'})
        previous_bytes = (directory / 'config.json').read_bytes()
        restarted = {**payload, 'preprocessing': payload['preprocessing'].replace('0x7f123abc', '0x7654def')}
        restarted['protocol_hash'] = m.digest_json(restarted)
        adopted = m.resolve_run_config(directory, restarted)
        self.assertEqual(adopted['protocol_hash'], legacy_hash)
        self.assertEqual(m.digest_json(payload), m.digest_json({k:v for k,v in restarted.items() if k != 'protocol_hash'}))
        self.assertEqual((directory / 'config.json').read_bytes(), previous_bytes)
        # Addresses must not be stripped out of unrelated scientific settings.
        self.assertNotEqual(m.digest_json({'setting': '0x123'}), m.digest_json({'setting': '0x456'}))
        for key, value in [('learning_rate', .002), ('engine_hash', 'different-code'),
                           ('preprocessing', 'Compose(Resize(256))'), ('environment', {'torch': 'changed'})]:
            with self.assertRaisesRegex(ValueError, 'Protocol changed.*' + key):
                m.resolve_run_config(directory, {**restarted, key: value})
