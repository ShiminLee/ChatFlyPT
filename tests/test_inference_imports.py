"""Serving saved readouts must not load the training-time sklearn dependency."""
import subprocess
import sys
from pathlib import Path


def test_server_and_saved_readout_modules_import_without_training_libraries():
    script='''
import sys
from importlib.abc import MetaPathFinder
class NoTrainingImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise AssertionError('Inference loaded training dependency: '+fullname)
sys.meta_path.insert(0, NoTrainingImports())
import chatflypt.server
import chatflypt.experiment
import chatflypt.sensory_readout
import chatflypt.state_readout
import chatflypt.feedback_readout
import chatflypt.unified_readout
import chatflypt.acoustic_readout
assert 'onnxruntime' not in sys.modules
assert 'h5py' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
