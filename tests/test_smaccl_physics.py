"""Physical consistency of the smaccl (inverse) and smaccl_dir (forward) kernels."""
import numpy as np
import pytest

from conftest import run_dir, run_inv

FWD = 'rtoa rtoa_0 dev_std Jr Juo3 Juh2o Jpre Jtaup'.split()
INV = 'rsurf rsurf_0 dev_std Jr Juo3 Juh2o Jpre Jtaup'.split()


def max_rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


def test_outputs_shape_dtype_finite(S_dir, S_inv, coeffs, inputs):
    fwd = run_dir(S_dir, coeffs, inputs)
    inv = run_inv(S_inv, coeffs, inputs, fwd[0])
    for name, arr in list(zip(FWD, fwd)) + list(zip(INV, inv)):
        assert arr.shape == inputs['rsurf'].shape, name
        assert arr.dtype == np.float32, name
        assert np.isfinite(arr).all(), name


def test_round_trip_brdf(S_dir, S_inv, coeffs, inputs):
    """smaccl(smaccl_dir(rsurf)) recovers rsurf with a BRDF surface."""
    rtoa = run_dir(S_dir, coeffs, inputs)[0]
    rsurf = run_inv(S_inv, coeffs, inputs, rtoa)[0]
    assert max_rel(rsurf, inputs['rsurf']) < 1e-5


def test_round_trip_without_brdf_coupling(S_dir, S_inv, coeffs, inputs):
    """rtoa_0 (forward, no BRDF coupling) inverts back through rsurf_0."""
    rtoa_0 = run_dir(S_dir, coeffs, inputs)[1]
    rsurf_0 = run_inv(S_inv, coeffs, inputs, rtoa_0)[1]
    assert max_rel(rsurf_0, inputs['rsurf']) < 1e-5


def test_lambertian_surface(S_dir, S_inv, coeffs, inputs):
    """With k1p = k2p = 0 the BRDF and Lambertian outputs coincide."""
    zero = np.zeros_like(inputs['k1p'])
    lamb = dict(inputs, k1p=zero, k2p=zero)
    fwd = run_dir(S_dir, coeffs, lamb)
    np.testing.assert_allclose(fwd[1], fwd[0], rtol=1e-6)
    inv = run_inv(S_inv, coeffs, lamb, fwd[0])
    np.testing.assert_allclose(inv[1], inv[0], rtol=1e-6)


def test_brdf_changes_toa(S_dir, coeffs, inputs):
    fwd = run_dir(S_dir, coeffs, inputs)
    assert not np.allclose(fwd[0], fwd[1])


def test_jr_forward_inverse_reciprocal(S_dir, S_inv, coeffs, inputs):
    """d rtoa / d rsurf  *  d rsurf / d rtoa  == 1."""
    fwd = run_dir(S_dir, coeffs, inputs)
    inv = run_inv(S_inv, coeffs, inputs, fwd[0])
    np.testing.assert_allclose(fwd[3] * inv[3], 1., rtol=1e-4)


@pytest.mark.parametrize('name, idx, tol', [('Juo3', 4, 1e-4), ('Juh2o', 5, 1e-4),
                                            ('Jpre', 6, 5e-3), ('Jtaup', 7, 5e-2)])
def test_jacobians_forward_vs_inverse(S_dir, S_inv, coeffs, inputs, name, idx, tol):
    """Implicit function theorem: d rtoa/dx = -(d rsurf/dx) / (d rsurf/d rtoa)."""
    fwd = run_dir(S_dir, coeffs, inputs)
    inv = run_inv(S_inv, coeffs, inputs, fwd[0])
    pred = -inv[idx] / inv[3]
    rel = np.abs(pred - fwd[idx]) / np.maximum(np.abs(fwd[idx]), 1e-8)
    assert float(np.median(rel)) < tol, name


def test_jr_vs_numerical_derivative(S_dir, coeffs, inputs):
    eps = 1e-3
    up = run_dir(S_dir, coeffs, inputs, rsurf=inputs['rsurf'] + eps)[0]
    dn = run_dir(S_dir, coeffs, inputs, rsurf=inputs['rsurf'] - eps)[0]
    Jr = run_dir(S_dir, coeffs, inputs)[3]
    np.testing.assert_allclose((up - dn) / (2 * eps), Jr, rtol=1e-2)


@pytest.mark.parametrize('var, idx', [('uo3', 4), ('uh2o', 5)])
def test_gas_jacobians_vs_numerical_derivative(S_dir, coeffs, inputs, var, idx):
    h = 0.02 * inputs[var]
    up = run_dir(S_dir, coeffs, inputs, **{var: inputs[var] + h})[0]
    dn = run_dir(S_dir, coeffs, inputs, **{var: inputs[var] - h})[0]
    J = run_dir(S_dir, coeffs, inputs)[idx]
    num = (up - dn) / (2 * h)
    rel = np.abs(num - J) / np.abs(J)
    assert float(np.median(rel)) < 2e-2, var


def test_pressure_aot_jacobians_sign(S_dir, coeffs, inputs):
    """More atmosphere (pressure, AOT) brightens dark surfaces at TOA."""
    dark = dict(inputs, rsurf=np.full_like(inputs['rsurf'], 0.02))
    fwd = run_dir(S_dir, coeffs, dark)
    assert (fwd[6] > 0).all() and (fwd[7] > 0).all()


# ----- aerosol-model ensemble -------------------------------------------------

def test_dev_std_single_member_is_zero(S_dir, coeffs, inputs):
    fwd = run_dir(S_dir, coeffs, inputs, iaero=inputs['iaero'][:1].copy())
    assert (fwd[2] == 0).all()


def test_dev_std_identical_members_is_zero(S_dir, coeffs, inputs):
    same = np.ascontiguousarray(np.stack([inputs['iaero'][0]] * 3))
    fwd = run_dir(S_dir, coeffs, inputs, iaero=same)
    assert (fwd[2] == 0).all()


@pytest.mark.parametrize('mode', ['dir', 'inv'])
def test_dev_std_distinct_members_positive(S_dir, S_inv, coeffs, inputs, mode):
    fwd = run_dir(S_dir, coeffs, inputs)
    out = fwd if mode == 'dir' else run_inv(S_inv, coeffs, inputs, fwd[0])
    assert (out[2] > 0).all()


@pytest.mark.parametrize('mode', ['dir', 'inv'])
def test_member0_outputs_independent_of_ensemble(S_dir, S_inv, coeffs, inputs, mode):
    """Only dev_std depends on members ia >= 1."""
    one = inputs['iaero'][:1].copy()
    rtoa = run_dir(S_dir, coeffs, inputs)[0]
    if mode == 'dir':
        full, single = run_dir(S_dir, coeffs, inputs), run_dir(S_dir, coeffs, inputs, iaero=one)
    else:
        full, single = run_inv(S_inv, coeffs, inputs, rtoa), run_inv(S_inv, coeffs, inputs, rtoa, iaero=one)
    for k in (0, 1, 3, 4, 5, 6, 7):
        np.testing.assert_array_equal(full[k], single[k])


def test_aerosol_model_selection(S_dir, coeffs, inputs):
    """Member 0 drives rtoa: changing its model changes rtoa."""
    a = run_dir(S_dir, coeffs, inputs)[0]
    other = inputs['iaero'][::-1].copy()   # member 0 now uses model 2
    b = run_dir(S_dir, coeffs, inputs, iaero=other)[0]
    assert not np.allclose(a, b)


@pytest.mark.parametrize('mode', ['dir', 'inv'])
def test_nbloop_does_not_change_results(S_dir, S_inv, coeffs, inputs, mode):
    """Extra NBLOOP runs repeat the reference case (no Monte-Carlo draw yet)."""
    d = inputs
    args = [d[k] for k in ('tetas', 'tetav', 'phis', 'phiv', 'uh2o', 'uo3', 'taup550', 'pression')]
    if mode == 'dir':
        f = lambda n: S_dir.run_dir(coeffs, *args, d['rsurf'], d['k1p'], d['k2p'], d['iaero'], NBLOOP=n)
    else:
        rtoa = run_dir(S_dir, coeffs, inputs)[0]
        f = lambda n: S_inv.run(coeffs, *args, rtoa, d['k1p'], d['k2p'], d['iaero'], NBLOOP=n)
    ref = f(0)
    for n in (1, 3):
        for a, b in zip(ref, f(n)):
            np.testing.assert_array_equal(a, b)
