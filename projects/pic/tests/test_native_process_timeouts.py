"""Real bridge/descendant shutdown, independently of Streamlit and NN imports."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIC))


class NativeProcessTimeouts(unittest.TestCase):
    def test_timeout_stops_descendant_processes(self):
        import psutil
        from epde_bench import pdecomp_methods as methods
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'slow_bridge.py'
            script.write_text(
                'import json,sys,time,subprocess\nfrom pathlib import Path\n'
                'request=json.loads(sys.argv[1])\n'
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                'Path(request["root"],"child.pid").write_text(str(child.pid))\n'
                'time.sleep(60)\n', encoding='utf-8')
            with patch.object(methods, 'source_dir', return_value=root), patch.object(methods, 'BRIDGE', script):
                with self.assertRaisesRegex(TimeoutError, 'time limit'):
                    methods._bridge({'method': 'deepmod'}, timeout=2)
            self.assertTrue((root / 'child.pid').is_file())
            pid = int((root / 'child.pid').read_text())
            # A killed orphan may remain a zombie until Linux init reaps it.
            # It cannot keep training or pipes open; pid_exists alone reports
            # it as present even though computation has stopped.
            try:
                child = psutil.Process(pid)
                self.assertTrue(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    unittest.main()
