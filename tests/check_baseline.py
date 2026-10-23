"""Numerical regression against the untouched sibling UniMatch-V2 model."""
import importlib.util
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import torch
from model.semseg.dpt import DPT


def main():
    torch.set_num_threads(2)
    source = Path(__file__).resolve().parents[2] / 'UniMatch-V2/model/semseg/dpt.py'
    spec = importlib.util.spec_from_file_location('original_dpt', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    kwargs = dict(encoder_size='small', nclass=3, features=16, out_channels=[8, 16, 32, 64])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline, modified = original.DPT(**kwargs).to(device).eval(), DPT(**kwargs).to(device).eval()
    modified.load_state_dict(baseline.state_dict())
    x = torch.randn(2, 3, 56, 56, device=device)
    with torch.no_grad():
        expected = baseline(x)
        actual, representation = modified(x, return_features=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert representation.shape == (2, 384, 4, 4)
        torch.manual_seed(42)
        expected = baseline(x, comp_drop=True)
        torch.manual_seed(42)
        actual = modified(x, comp_drop=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print('Original logits and complementary perturbation: bitwise identical; feature output: PASS')


if __name__ == '__main__':
    main()