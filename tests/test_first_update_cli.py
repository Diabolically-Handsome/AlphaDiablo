"""Opt-in capture boundaries at the actual training CLI; no model or game."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train'), str(ROOT / 'python')]
import train_ppo as training


def args(**overrides):
    result = dict(diagnostic_rollout='first-update-v1', worker=True,
                  algo='mppo', device='cpu', artifact_scope='candidate',
                  resource_warm_start='explicit/manifest.json', resume_from=None,
                  total_steps=2048, n_steps=512, num_envs=4,
                  calib_probes='', calib_record_only=False)
    result.update(overrides)
    return SimpleNamespace(**result)


class FirstUpdateCliTests(unittest.TestCase):
    def test_disabled_requires_no_experimental_fields_or_callback_import(self):
        empty = SimpleNamespace()
        training._validate_first_update_diagnostic_args(empty)
        with patch.dict(sys.modules, {'rollout_diagnostics': None}):
            self.assertIsNone(training._first_update_diagnostic_callback(empty, '/unused', 'a' * 64))

    def test_only_explicit_complete_fresh_candidate_is_accepted(self):
        training._validate_first_update_diagnostic_args(args())
        invalid = [dict(diagnostic_rollout='unknown'), dict(worker=False),
                   dict(algo='ppo'), dict(device='cuda'), dict(artifact_scope='production'),
                   dict(artifact_scope='development'), dict(resource_warm_start=None),
                   dict(resume_from='previous.zip'), dict(total_steps=8192),
                   dict(total_steps=2047), dict(calib_probes='2048'),
                   dict(calib_record_only=True)]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                training._validate_first_update_diagnostic_args(args(**override))

    def test_mount_passes_exact_run_and_implementation_identity(self):
        constructor = Mock(return_value=object())
        module = SimpleNamespace(FirstUpdateDiagnosticCallback=constructor)
        with patch.dict(sys.modules, {'rollout_diagnostics': module}):
            result = training._first_update_diagnostic_callback(args(), '/candidate/run', 'b' * 64)
        constructor.assert_called_once_with('/candidate/run', 'b' * 64)
        self.assertIs(result, constructor.return_value)

    def test_real_parser_defaults_disabled_and_accepts_explicit_flag(self):
        class StopBeforeAnyIO(Exception):
            pass
        for extra, expected in [([], 'disabled'),
                                (['--diagnostic-rollout', 'first-update-v1'], 'first-update-v1')]:
            captured = []
            def stop(parsed):
                captured.append(parsed)
                raise StopBeforeAnyIO()
            with patch.object(sys, 'argv', ['train_ppo.py', *extra]), \
                    patch.object(training, '_validate_args', side_effect=stop), \
                    self.assertRaises(StopBeforeAnyIO):
                training._main(SimpleNamespace())
            self.assertEqual(captured[0].diagnostic_rollout, expected)

    def flow_fixture(self):
        path = ROOT / 'tests/test_training_completion_diagnostics.py'
        spec = importlib.util.spec_from_file_location('completion_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fixture = module.FinalizationIntegrationTests()
        fixture.setUpClass()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_capture_constructor_failure_preserves_original_error_and_cleanup(self):
        fixture = self.flow_fixture()
        error = OSError('diagnostic directory unavailable')
        with patch.object(training, '_first_update_diagnostic_callback', side_effect=error):
            result = fixture.run_case()
        self.assertIs(result.caught, error)
        result.model.learn.assert_not_called()
        self.assertEqual(result.events, ['callback.close', 'resources.close'])
        self.assertEqual(result.status['publication_status'], 'TRAINING_ERROR')

    def test_main_passes_capture_last_without_changing_publication(self):
        fixture = self.flow_fixture()
        capture = object()
        with patch.object(training, '_first_update_diagnostic_callback', return_value=capture):
            result = fixture.run_case()
        self.assertIsNone(result.caught)
        self.assertIs(result.model.learn.call_args.kwargs['callback'][-1], capture)
        self.assertEqual(result.status['publication_status'], 'PRODUCTION_CANDIDATE')
        self.assertEqual(result.events, ['callback.close', 'candidate.save', 'resources.close'])


if __name__ == '__main__':
    unittest.main()
