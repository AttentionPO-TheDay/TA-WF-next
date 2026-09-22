import importlib.util
from pathlib import Path
import numpy as np
import torch

P=Path(__file__).resolve().parents[1]/"run_interference_diagnostic.py"
S=importlib.util.spec_from_file_location("diag",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

def test_model_checkpoint_and_scope():
    model,names=M.make_model(torch.device("cpu"))
    assert len(names)==20
    assert all(("block.1." in n or "block.4." in n or n.startswith("classifier.2.") or n.startswith("classifier.6.")) for n in names)
    assert M.sha256_file(M.CHECKPOINT)==M.EXPECTED_CHECKPOINT_SHA256

def test_metrics_transitions_and_macro():
    y=np.array([0,1,2,2,2]); b=np.array([[3,1,0],[0,3,1],[0,1,3],[3,1,0],[3,1,0.]],float); u=np.array([[1,3,0],[0,3,1],[0,1,3],[0,3,1],[0,1,3.]],float)
    q=M.metrics(y,b,u)
    assert q["wrong_to_correct"]==1 and q["correct_to_wrong"]==1
    assert q["wrong_to_different_wrong"]==1 and q["unchanged_correct"]==2
    assert 0<=M.macro_f1(y,u.argmax(1))<=1

def test_adaptation_does_not_accept_labels():
    assert "targets" not in M.adapt.__code__.co_varnames
    assert "y" not in M.adapt.__code__.co_varnames

def test_spearman():
    assert abs(M.spearman([1,2,3],[3,2,1])+1)<1e-12
