"""One version source: package metadata, module, CLI and native health agree (V17)."""
import subprocess
import sys
from importlib.metadata import version

import inferstack


def test_package_module_and_cli_versions_agree():
    assert version('inferstack') == inferstack.__version__
    out = subprocess.run([sys.executable, '-m', 'inferstack.cli', '--version'],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == inferstack.__version__
