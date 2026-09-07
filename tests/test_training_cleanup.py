"""Owned training-worker cleanup only: no game, model, or optimizer import."""
import ast
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import signal
import time
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / 'train/train_ppo.py'
tree = ast.parse(SOURCE.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
            and n.name == '_TrainingResources')
namespace = {'json': json}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
TrainingResources = namespace['_TrainingResources']


class Process:
    def __init__(self, pid=123, *, alive=True, ignore_term=False):
        self.pid = pid
        self.alive = alive
        self.ignore_term = ignore_term
        self.calls = []

    def is_alive(self):
        self.calls.append('is_alive')
        return self.alive

    def terminate(self):
        self.calls.append('terminate')
        if not self.ignore_term:
            self.alive = False

    def kill(self):
        self.calls.append('kill')
        self.alive = False

    def join(self, timeout=None):
        self.calls.append(('join', timeout))
        if timeout is None:
            raise AssertionError('join must be bounded')


class Vec:
    def __init__(self, processes=(), remotes=(), error=None):
        self.processes = list(processes)
        self.remotes = list(remotes)
        self.error = error
        self.closes = 0

    def close(self):
        self.closes += 1
        if self.error is not None:
            raise self.error


def _idle_child(connection, ignore_term):
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    connection.send('ready')
    connection.close()
    while True:
        signal.pause()


class TrainingCleanupTests(unittest.TestCase):
    def close(self, env):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            TrainingResources._close_vec_env(env)
        return output.getvalue()

    def test_normal_close_does_not_touch_process_or_pipe_handles(self):
        process, remote = Process(), mock.Mock()
        env = Vec([process], [remote])
        self.assertEqual(self.close(env), '')
        self.assertEqual(env.closes, 1)
        self.assertEqual(process.calls, [])
        remote.close.assert_not_called()

    def test_eof_oserror_and_timeout_reap_remaining_owned_workers(self):
        for error in (EOFError(), OSError('broken pipe'), TimeoutError('deadline')):
            with self.subTest(error=type(error).__name__):
                live, dead = Process(101), Process(102, alive=False)
                remote = mock.Mock()
                text = self.close(Vec([live, dead], [remote], error))
                self.assertIn(type(error).__name__, text)
                self.assertIn('"remaining": []', text)
                self.assertIn('terminate', live.calls)
                self.assertNotIn('kill', live.calls)
                self.assertNotIn('terminate', dead.calls)
                self.assertTrue(any(isinstance(x, tuple) for x in dead.calls))
                remote.close.assert_called_once_with()

    def test_close_mutation_cannot_erase_owned_handle_snapshot(self):
        process, remote = Process(), mock.Mock()
        env = Vec([process], [remote])

        def broken_close():
            env.processes.clear()
            env.remotes.clear()
            raise EOFError()

        env.close = broken_close
        self.close(env)
        self.assertIn('terminate', process.calls)
        remote.close.assert_called_once_with()

    def test_no_owned_processes_does_not_enumerate_or_signal_other_children(self):
        remote = mock.Mock()
        with mock.patch('multiprocessing.active_children', side_effect=AssertionError('global lookup')):
            text = self.close(Vec([], [remote], EOFError()))
        self.assertIn('EOFError', text)
        remote.close.assert_not_called()

    def test_real_term_ignoring_worker_is_killed_but_unrelated_child_survives(self):
        ctx = multiprocessing.get_context('spawn')
        processes, pipes = [], []
        try:
            for ignore_term in (True, False):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_idle_child, args=(child, ignore_term))
                process.start()
                child.close()
                processes.append(process)
                pipes.append(parent)
                self.assertTrue(parent.poll(10), 'short worker failed to start')
                self.assertEqual(parent.recv(), 'ready')
            start = time.monotonic()
            text = self.close(Vec([processes[0]], [pipes[0]], EOFError()))
            self.assertLess(time.monotonic() - start, 7)
            self.assertFalse(processes[0].is_alive())
            self.assertEqual(processes[0].exitcode, -signal.SIGKILL)
            self.assertIn(str(processes[0].pid), text)
            self.assertIn('"remaining": []', text)
            self.assertTrue(processes[1].is_alive())
        finally:
            # These are test-owned children, never a global child search.
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join(5)
            for connection in pipes:
                connection.close()

    def test_join_deadline_is_shared_not_restarted_per_worker(self):
        clock = [0.0]
        processes = [Process(i, ignore_term=True) for i in range(3)]
        seen = []
        for process in processes:
            def join(timeout=None):
                self.assertIsNotNone(timeout)
                seen.append(timeout)
                clock[0] += min(0.75, timeout)
            process.join = join
        with mock.patch('time.monotonic', side_effect=lambda: clock[0]):
            result = TrainingResources._reap_failed_vec_env(processes, [])
        self.assertEqual(seen, [2.0, 1.25, 0.5, 2.0, 1.25, 0.5])
        self.assertEqual(clock[0], 4.0)
        self.assertEqual(result['remaining'], [])

    def test_failure_of_one_operation_does_not_skip_other_workers_or_kill_fallback(self):
        first, second = Process(101), Process(102)
        first.terminate = mock.Mock(side_effect=OSError('terminate failed'))
        remote = mock.Mock()
        remote.close.side_effect = OSError('pipe close failed')
        result = TrainingResources._reap_failed_vec_env([first, second], [remote])
        self.assertFalse(first.alive or second.alive)
        self.assertIn('kill', first.calls)
        self.assertIn('terminate', second.calls)
        self.assertEqual(len(result['errors']), 2)

    def test_alarm_timeout_restores_host_handler_and_alarm_before_fallback(self):
        previous = object()
        handler = [previous]
        events = []

        def set_handler(sig, value):
            handler[0] = value
            events.append(('handler', value))

        env = Vec([Process()])
        env.close = lambda: handler[0]()

        def recover(processes, remotes):
            self.assertIs(handler[0], previous)
            self.assertEqual(events[-1], ('alarm', 7))
            return {'remaining': []}

        with mock.patch('signal.getsignal', return_value=previous), \
                mock.patch('signal.signal', side_effect=set_handler), \
                mock.patch('signal.alarm', side_effect=lambda n: events.append(('alarm', n)) or (7 if len(events) == 1 else 0)), \
                mock.patch.object(TrainingResources, '_reap_failed_vec_env', side_effect=recover) as fallback:
            self.assertIn('TimeoutError', self.close(env))
        fallback.assert_called_once()

    def test_keyboard_interrupt_is_not_swallowed_or_reclassified_as_close_failure(self):
        env = Vec([Process()], error=KeyboardInterrupt())
        before = signal.getsignal(signal.SIGALRM)
        with mock.patch.object(TrainingResources, '_reap_failed_vec_env') as fallback:
            with self.assertRaises(KeyboardInterrupt):
                self.close(env)
        fallback.assert_not_called()
        self.assertIs(signal.getsignal(signal.SIGALRM), before)

    def test_signal_during_fallback_propagates_and_owner_still_releases_lock(self):
        process = Process()
        process.terminate = mock.Mock(side_effect=KeyboardInterrupt())
        resources = TrainingResources()
        lock = resources.run_lock = mock.Mock()
        resources.vec_env = Vec([process], error=EOFError())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            resources.close()
        lock.close.assert_called_once_with()

    def test_original_training_exception_survives_failed_cleanup_and_close_is_idempotent(self):
        process = Process()
        process.terminate = mock.Mock(side_effect=OSError('term'))
        process.kill = mock.Mock(side_effect=OSError('kill'))
        process.join = mock.Mock(side_effect=OSError('join'))
        resources = TrainingResources()
        lock = resources.run_lock = mock.Mock()
        env = resources.vec_env = Vec([process], error=EOFError())
        original = RuntimeError('original worker error')
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError) as caught:
                try:
                    raise original
                finally:
                    resources.close()
            resources.close()
        self.assertIs(caught.exception, original)
        self.assertIn('"remaining": [123]', output.getvalue())
        self.assertEqual(env.closes, 1)
        lock.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
