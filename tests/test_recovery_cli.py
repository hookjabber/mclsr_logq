from pathlib import Path
import subprocess
import sys


def test_recovery_cli_exposes_explicit_continuation_and_launcher():
    script = Path(__file__).resolve().parents[1] / 'scripts/recover_protocol_study.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, check=True)
    assert '--previous-recovery' in result.stdout
    assert '--launcher' in result.stdout
