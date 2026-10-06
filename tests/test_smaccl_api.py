"""Python API of Smaccl: mode selector, launch dispatch, input handling."""
import numpy as np
import pytest

from conftest import run_dir, run_inv


def test_default_mode_is_inverse(S_inv):
    assert S_inv.mode == 'smaccl'


def test_invalid_mode_rejected(smaccl_module):
    with pytest.raises(ValueError):
        smaccl_module.Smaccl(platform='CPU', mode='both')


def test_run_refused_in_forward_mode(S_dir, coeffs, inputs):
    with pytest.raises(RuntimeError, match="mode='smaccl'"):
        run_inv(S_dir, coeffs, inputs, inputs['rsurf'])


def test_run_dir_refused_in_inverse_mode(S_inv, coeffs, inputs):
    with pytest.raises(RuntimeError, match="mode='smaccl_dir'"):
        run_dir(S_inv, coeffs, inputs)


def test_launch_dispatches_to_selected_kernel(S_dir, S_inv, coeffs, inputs):
    d = inputs
    args = [d[k] for k in ('tetas', 'tetav', 'phis', 'phiv', 'uh2o', 'uo3', 'taup550', 'pression')]
    fwd = run_dir(S_dir, coeffs, inputs)
    fwd_l = S_dir.launch(coeffs, *args, d['rsurf'], d['k1p'], d['k2p'], d['iaero'])
    inv = run_inv(S_inv, coeffs, inputs, fwd[0])
    inv_l = S_inv.launch(coeffs, *args, fwd[0], d['k1p'], d['k2p'], d['iaero'])
    for a, b in zip(fwd, fwd_l):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inv, inv_l):
        np.testing.assert_array_equal(a, b)


def test_run_dir_casts_iaero_to_int16(S_dir, coeffs, inputs):
    ref = run_dir(S_dir, coeffs, inputs)
    out = run_dir(S_dir, coeffs, inputs, iaero=inputs['iaero'].astype(np.int64))
    for a, b in zip(ref, out):
        np.testing.assert_array_equal(a, b)


def test_run_dir_returns_eight_arrays(S_dir, coeffs, inputs):
    assert len(run_dir(S_dir, coeffs, inputs)) == 8


def test_run_returns_eight_arrays(S_dir, S_inv, coeffs, inputs):
    rtoa = run_dir(S_dir, coeffs, inputs)[0]
    assert len(run_inv(S_inv, coeffs, inputs, rtoa)) == 8


def test_repeated_calls_are_deterministic(S_dir, coeffs, inputs):
    a, b = run_dir(S_dir, coeffs, inputs), run_dir(S_dir, coeffs, inputs)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
