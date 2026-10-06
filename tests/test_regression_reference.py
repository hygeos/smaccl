"""Frozen reference outputs of smaccl and smaccl_dir.

`data/reference_outputs.npz` holds fixed inputs and the outputs of both kernels
produced by a validated version of the code. Any change of the numerical
results makes this test fail.

After an intended change of the physics, regenerate the reference with:

    SMACCL_UPDATE_REFERENCE=1 pytest tests/test_regression_reference.py

and commit the new .npz together with the change.
"""
import os

import numpy as np
import pytest

from conftest import ANCILLARY, make_inputs

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference_outputs.npz')
FWD = 'rtoa rtoa_0 dev_std Jr Juo3 Juh2o Jpre Jtaup'.split()
INV = 'rsurf rsurf_0 dev_std Jr Juo3 Juh2o Jpre Jtaup'.split()
INPUTS = ANCILLARY + ('rsurf', 'k1p', 'k2p', 'iaero')
# float32 kernels: allow for compiler / device rounding differences (CPU vs GPU)
RTOL, ATOL = 1e-4, 1e-6


def compute(S_dir, S_inv, coeffs, d):
    anc = [d[k] for k in ANCILLARY]
    fwd = S_dir.run_dir(coeffs, *anc, d['rsurf'], d['k1p'], d['k2p'], d['iaero'])
    # invert a TOA signal that differs from fwd[0] so the inverse is not a mere round trip
    rtoa = np.ascontiguousarray(fwd[0] * np.float32(1.02))
    inv = S_inv.run(coeffs, *anc, rtoa, d['k1p'], d['k2p'], d['iaero'])
    out = {'dir_' + k: v for k, v in zip(FWD, fwd)}
    out.update({'inv_' + k: v for k, v in zip(INV, inv)})
    out['inv_input_rtoa'] = rtoa
    return out


def test_reference_outputs(S_dir, S_inv, coeffs):
    if os.environ.get('SMACCL_UPDATE_REFERENCE'):
        d = make_inputs(seed=42)
        out = compute(S_dir, S_inv, coeffs, d)
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        np.savez_compressed(REF, coeffs=coeffs, **{'in_' + k: d[k] for k in INPUTS}, **out)
        pytest.skip('reference regenerated: ' + REF)

    assert os.path.exists(REF), 'missing reference file; see module docstring'
    ref = np.load(REF)
    d = {k: ref['in_' + k] for k in INPUTS}
    out = compute(S_dir, S_inv, ref['coeffs'], d)
    failures = []
    for k, v in out.items():
        try:
            np.testing.assert_allclose(v, ref[k], rtol=RTOL, atol=ATOL, err_msg=k)
        except AssertionError as e:
            failures.append(str(e).strip().splitlines()[0] + ' ' + k)
    assert not failures, 'outputs differ from the reference:\n' + '\n'.join(failures)
