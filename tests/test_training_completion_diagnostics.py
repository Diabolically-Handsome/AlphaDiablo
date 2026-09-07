"""Publication reporting/retention integration, without environment or learning."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train'), str(ROOT / 'python')]
import train_ppo as training


def report(*, boundary=True, exact=True, migration=True, pg=False, asymmetric=True):
    checks = dict(rollout_boundary=boundary, exact_target=exact,
                  migration=migration, worker_pg=pg, asymmetric=asymmetric)
    return {'schema': 'diablogym-training-completion/1',
            'publication_eligible': all(checks.values()), 'checks': checks,
            'failed_checks': [k for k, v in checks.items() if not v],
            'counters': {'num_timesteps': 2048, 'target_global_steps': 2048,
                'last_completed_ppo_rollout_steps': 2048,
                'ppo_optimizer_steps_completed': 80,
                'actor_optimizer_steps_completed': 80,
                'worker_onpolicy_pg_joint_rollouts': 1,
                'worker_onpolicy_pg_qualifying_rollouts': int(pg)}}


class CompletionReportTests(unittest.TestCase):
    def model(self):
        return types.SimpleNamespace(
            num_timesteps=2048, rollout_buffer=types.SimpleNamespace(full=True),
            _calib_tripped=False, _last_completed_ppo_rollout_steps=2048,
            _ppo_optimizer_steps_completed=80, _actor_optimizer_steps_completed=80,
            _worker_onpolicy_pg_joint_rollouts=1, _worker_onpolicy_pg_qualifying_rollouts=0,
            _critic_warmup_start_timesteps=None, _resource_warm_start_receipt={},
            _assert_critic_migration_contract=Mock())

    def test_quality_refusal_names_gate_without_claiming_unconsumed_update(self):
        model = self.model()
        with patch.object(training, '_asymmetric_worker_deployment_evidence_complete', return_value=True) as deployment, \
                patch('leashed_ppo.worker_onpolicy_pg_audit_complete', return_value=False) as pg:
            with self.assertRaises(training._TrainingCompletionRejected) as raised:
                training._require_exact_training_completion(model, 2048)
        failure = raised.exception
        self.assertEqual(failure.report, report())
        self.assertIn('训练更新已完成', str(failure))
        self.assertNotIn('未精确停在', str(failure))
        deployment.assert_called_once_with(model)
        pg.assert_called_once_with(model)
        model._assert_critic_migration_contract.assert_called_once_with()

    def test_report_preserves_independent_multiple_failures(self):
        model = self.model()
        model.rollout_buffer.full = False
        model.num_timesteps = 2047
        model._assert_critic_migration_contract.side_effect = ValueError('lineage')
        with patch.object(training, '_asymmetric_worker_deployment_evidence_complete', return_value=False), \
                patch('leashed_ppo.worker_onpolicy_pg_audit_complete', return_value=False):
            got = training._training_completion_report(model, 2048)
        self.assertEqual(got['failed_checks'],
                         ['rollout_boundary', 'exact_target', 'migration', 'worker_pg', 'asymmetric'])
        self.assertFalse(got['publication_eligible'])

    def test_success_returns_structured_evidence_without_changing_eligibility(self):
        model = self.model()
        model._worker_onpolicy_pg_qualifying_rollouts = 1
        with patch.object(training, '_asymmetric_worker_deployment_evidence_complete', return_value=True), \
                patch('leashed_ppo.worker_onpolicy_pg_audit_complete', return_value=True):
            got = training._require_exact_training_completion(model, 2048)
            self.assertTrue(training._is_exact_training_completion(model, 2048))
        self.assertEqual(got, report(pg=True))

    def test_original_frozen_predicate_agrees_across_failure_combinations(self):
        frozen = Path('/home/laure/r20_sustain_20260904/candidate-earned-v2/train/train_ppo.py')
        payload = frozen.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         '8dce3c4c776fa349c6338117aa795ccd91af2dc878b97f3447f3529628863915')
        tree = ast.parse(payload)
        original = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_is_exact_training_completion')
        original.name = 'original_completion'
        namespace = dict(training.__dict__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
                     str(frozen), 'exec'), namespace)
        for steps in (2047, 2048, 2049):
            for full in (False, True):
                for migration_valid in (False, True):
                    for pg_valid in (False, True):
                        for asymmetric_valid in (False, True):
                            model = self.model()
                            model.num_timesteps = steps
                            model.rollout_buffer.full = full
                            if not migration_valid:
                                model._assert_critic_migration_contract.side_effect = ValueError('lineage')
                            with self.subTest(steps=steps, full=full, migration=migration_valid,
                                              pg=pg_valid, asymmetric=asymmetric_valid), \
                                    patch.object(training, '_asymmetric_worker_deployment_evidence_complete', return_value=asymmetric_valid), \
                                    patch('leashed_ppo.worker_onpolicy_pg_audit_complete', return_value=pg_valid):
                                namespace['_asymmetric_worker_deployment_evidence_complete'] = \
                                    training._asymmetric_worker_deployment_evidence_complete
                                self.assertEqual(namespace['original_completion'](model, 2048),
                                                 training._is_exact_training_completion(model, 2048))

    def test_ineligible_boundary_does_not_serialize_or_inspect_implementation(self):
        model = self.model()
        for bad in (report(boundary=False), report(exact=False), report(migration=False)):
            with patch.object(training, '_implementation_bundle_sha256') as identity:
                got = training._retain_refused_training_diagnostic(model, Path('/unused'), bad, 'a' * 64)
            self.assertEqual(got['status'], 'NOT_ELIGIBLE')
            identity.assert_not_called()

    def test_identity_drift_is_rejected_before_archive(self):
        model = self.model()
        with patch.object(training, '_implementation_bundle_sha256', return_value='b' * 64), \
                patch('training_diagnostics.archive_refused_training') as archive:
            with self.assertRaisesRegex(ValueError, '实现/引擎/游戏内容发生漂移'):
                training._retain_refused_training_diagnostic(model, Path('/unused'), report(), 'a' * 64)
        archive.assert_not_called()


class FinalizationIntegrationTests(unittest.TestCase):
    """Execute the actual main tail AST, replacing only training and I/O owners."""
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'train/train_ppo.py'
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_main')
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'callback' for t in n.targets)
                     and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == 'EpisodeJsonlCallback')
        wrapper = ast.FunctionDef(name='run_tail', args=ast.arguments(posonlyargs=[], args=[],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=copy.deepcopy(main.body[start:]),
            decorator_list=[])
        cls.code = compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), 'exec')

    def run_case(self, *, failure=None, learn_error=None, diagnostic_error=None,
                 scope='candidate', use_real_ineligible_retainer=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        run_dir = Path(directory.name)
        events = []
        model = types.SimpleNamespace(num_timesteps=2048, _resource_warm_start_receipt={})
        model.learn = Mock(side_effect=learn_error)
        callback = types.SimpleNamespace(close=lambda: events.append('callback.close'))
        resources = types.SimpleNamespace(close=lambda: events.append('resources.close'))
        completion = Mock(side_effect=failure, return_value=report(pg=True))
        retention = Mock(side_effect=diagnostic_error,
                         return_value={'status': 'SAVED', 'path': 'training_diagnostic.zip', 'sha256': 'd' * 64})
        if use_real_ineligible_retainer:
            retention = Mock(wraps=training._retain_refused_training_diagnostic)
        def save(model, path):
            events.append('candidate.save')
            path.write_bytes(b'fixture only, not a model')
        namespace = dict(training.__dict__)
        namespace.update(model=model, run_dir=run_dir, target_global_steps=2048,
            args=types.SimpleNamespace(total_steps=2048, resume_from=None, artifact_scope=scope),
            config={}, resources=resources, implementation_sha256='a' * 64,
            EpisodeJsonlCallback=lambda *args: callback, ckpt=object(), bc_aux_bank=None,
            _require_exact_training_completion=completion,
            _retain_refused_training_diagnostic=retention,
            _implementation_bundle_sha256=lambda: 'a' * 64, _atomic_save_model=save)
        for name in ('curriculum_cb', 'unfreeze_cb', 'sentinel_cb', 'r13_dive_audit_cb',
                     'prefix_audit_cb', 'dry_cb', 'distill_ce_cb', 'drywin_cb'):
            namespace[name] = None
        exec(self.code, namespace)
        caught = None
        try:
            namespace['run_tail']()
        except BaseException as exc:
            caught = exc
        status = json.loads((run_dir / 'status.json').read_text())
        return types.SimpleNamespace(model=model, retention=retention, completion=completion,
            events=events, caught=caught, status=status, run_dir=run_dir)

    def test_normal_candidate_publication_is_unchanged_and_never_archives(self):
        result = self.run_case()
        self.assertIsNone(result.caught)
        self.assertEqual(result.status['publication_status'], 'PRODUCTION_CANDIDATE')
        self.assertTrue(result.status['learn_returned_normally'])
        self.assertTrue(result.status['model_production_candidate'])
        self.assertTrue(result.status['training_completion']['publication_eligible'])
        result.retention.assert_not_called()
        self.assertEqual(result.events, ['callback.close', 'candidate.save', 'resources.close'])

    def test_quality_refusal_retains_diagnostic_and_original_nonzero_cause(self):
        failure = training._TrainingCompletionRejected(report())
        result = self.run_case(failure=failure)
        self.assertIs(result.caught, failure)
        result.retention.assert_called_once()
        self.assertEqual(result.status['publication_status'], 'TRAINING_ERROR')
        self.assertFalse(result.status['model_production_candidate'])
        self.assertIsNone(result.status['model_sha256'])
        self.assertEqual(result.status['training_diagnostic']['status'], 'SAVED')
        self.assertEqual(result.status['training_completion'], failure.report)
        self.assertTrue(result.status['learn_returned_normally'])
        self.assertEqual(result.events, ['callback.close', 'resources.close'])
        self.assertFalse((result.run_dir / 'model_candidate.zip').exists())

    def test_diagnostic_io_error_does_not_replace_refusal_or_skip_cleanup(self):
        failure = training._TrainingCompletionRejected(report())
        result = self.run_case(failure=failure, diagnostic_error=OSError('disk full'))
        self.assertIs(result.caught, failure)
        self.assertEqual(result.status['training_diagnostic'], {'status': 'FAILED', 'error': 'OSError: disk full'})
        self.assertEqual(result.events, ['callback.close', 'resources.close'])
        self.assertIsNone(result.status['model_sha256'])

    def test_incomplete_return_is_reported_but_does_not_save(self):
        failure = training._TrainingCompletionRejected(report(boundary=False))
        result = self.run_case(failure=failure, use_real_ineligible_retainer=True)
        self.assertIs(result.caught, failure)
        self.assertEqual(result.status['training_diagnostic']['status'], 'NOT_ELIGIBLE')
        self.assertEqual(result.events, ['callback.close', 'resources.close'])

    def test_collection_error_has_no_final_probe_or_diagnostic(self):
        error = RuntimeError('collection failed')
        result = self.run_case(learn_error=error)
        self.assertIs(result.caught, error)
        result.completion.assert_not_called()
        result.retention.assert_not_called()
        self.assertFalse(result.status['learn_returned_normally'])
        self.assertNotIn('training_completion', result.status)
        self.assertNotIn('training_diagnostic', result.status)
        self.assertEqual(result.events, ['callback.close', 'resources.close'])

    def test_interruption_does_not_attempt_diagnostic_and_still_cleans(self):
        error = KeyboardInterrupt()
        result = self.run_case(learn_error=error)
        self.assertIs(result.caught, error)
        result.retention.assert_not_called()
        self.assertEqual(result.events, ['callback.close', 'resources.close'])

    def test_production_refusal_does_not_opt_into_experimental_retention(self):
        failure = training._TrainingCompletionRejected(report())
        result = self.run_case(failure=failure, scope='production')
        self.assertIs(result.caught, failure)
        result.retention.assert_not_called()
        self.assertNotIn('training_diagnostic', result.status)


if __name__ == '__main__':
    unittest.main()
