"""The optimizer lesson must not touch EPDE globals in the Streamlit process."""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PIC = Path(__file__).resolve().parents[1]
APP = PIC / 'app'
sys.path[:0] = [str(APP), str(PIC), str(PIC.parent.parent)]


class OptimizerProcessTests(unittest.TestCase):
    def test_button_delegates_real_optimizer_to_child_process(self):
        from streamlit.testing.v1 import AppTest
        from epde_bench import optimizer_demo
        app = AppTest.from_file(str(APP / 'views/9_How_EPDE_works.py'), default_timeout=90).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        with patch.object(optimizer_demo, 'run_optimizer_demo',
                          side_effect=AssertionError('EPDE must not run in the Streamlit process')) as parent_run:
            next(b for b in app.button if b.label == 'Run short evolution').click().run()
        parent_run.assert_not_called()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertTrue(app.dataframe)
        self.assertEqual(list(app.dataframe[0].value['Epoch']), [1, 2, 3])
        self.assertIn('u_{tt}', app.latex[-1].value)

    def test_child_failure_reports_log_and_removes_temporary_files(self):
        from support.optimizer_process import run_optimizer_subprocess
        paths = []
        def fail(command, **kwargs):
            paths.append(Path(command[-1]))
            kwargs['stdout'].write('child execution failed\n')
            return subprocess.CompletedProcess(command, 1)
        with patch('support.optimizer_process.subprocess.run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'child execution failed'):
                run_optimizer_subprocess()
        self.assertTrue(paths)
        self.assertFalse(paths[0].parent.exists())

    def test_timeout_is_bounded_and_removes_temporary_files(self):
        from support.optimizer_process import run_optimizer_subprocess
        paths = []
        def expire(command, **kwargs):
            paths.append(Path(command[-1]))
            self.assertEqual(kwargs['timeout'], 60)
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        with patch('support.optimizer_process.subprocess.run', side_effect=expire):
            with self.assertRaisesRegex(RuntimeError, '60 seconds'):
                run_optimizer_subprocess()
        self.assertFalse(paths[0].parent.exists())


if __name__ == '__main__':
    unittest.main()
