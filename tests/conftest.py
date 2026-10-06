"""Shared fixtures for the smaccl regression tests.

The tests run the OpenCL kernels on the CPU through pocl ("Portable Computing
Language"), the platform Smaccl selects for platform='CPU'. They are skipped
when pyopencl or pocl is not available.

SMAC coefficients are synthetic but physically plausible: the tests check the
consistency of the code (forward / inverse round trip, Jacobians, ensemble
statistics, frozen reference outputs), not the radiometric accuracy of a sensor.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# test_modis_vito.py needs the `core` package and real MODIS / MERRA2 data on
# the processing servers; do not let it break collection elsewhere.
try:
    import core  # noqa: F401
except ImportError:
    collect_ignore = ['test_modis_vito.py']

POCL = 'Portable Computing Language'
NBAND, NMOD = 2, 3
NZ, XBLOCK, XGRID = 2, 4, 8


@pytest.fixture(scope='session')
def smaccl_module():
    cl = pytest.importorskip('pyopencl')
    try:
        platforms = [p.name for p in cl.get_platforms()]
    except cl.Error:
        platforms = []
    if POCL not in platforms:
        pytest.skip('pocl OpenCL platform not available (found: {})'.format(platforms))
    from smaccl import smaccl
    return smaccl


@pytest.fixture(scope='session')
def S_inv(smaccl_module):
    """Smaccl instance in the default (inverse) mode."""
    return smaccl_module.Smaccl(platform='CPU')


@pytest.fixture(scope='session')
def S_dir(smaccl_module):
    """Smaccl instance in forward (smaccl_dir) mode."""
    return smaccl_module.Smaccl(platform='CPU', mode='smaccl_dir')


def make_coeffs(type_coeff_reduced, nband=NBAND, nmod=NMOD):
    """Synthetic SMAC coefficients, one set per (band, aerosol model)."""
    names = np.dtype(type_coeff_reduced).names
    ca = np.zeros((nband, nmod), dtype=type_coeff_reduced)
    for ib in range(nband):
        for im in range(nmod):
            vals = dict(
                ah2o=-0.01 * (ib + 1), nh2o=0.6, ao3=-0.08, no3=1.0,
                ao2=-0.001, no2=0.5, po2=1.0,
                aco2=-0.0005, nco2=0.6, pco2=1.0, ach4=-0.0005, nch4=0.6, pch4=1.0,
                aco=-0.0005, nco=0.6, pco=1.0, pno2=1.0,
                a0s=0.08, a1s=0.15 + 0.03 * im, a2s=-0.05, a3s=0.0,
                a0T=0.98, a1T=-0.12 - 0.02 * im, a2T=-0.08, a3T=0.0,
                taur=0.05 / (ib + 1), a0taup=0.0, a1taup=0.9 - 0.1 * ib,
                wo=0.93 - 0.03 * im, gc=0.65 + 0.03 * im,
                a0P=3.0, a1P=-0.06, a2P=5e-4, a3P=-1.5e-6, a4P=1e-9,
                Resr2=0.001,
                f1d0=-1.0, f1d1=0.3, f1d2=0.1, f2d0=0.05, f2d1=0.1, f2d2=0.4,
                f1b0=-1.1, f1b1=0.2, f1b2=0.1, f2b0=0.04, f2b1=0.1, f2b2=0.3)
            for k, v in vals.items():
                if k in names:
                    ca[ib, im][k] = v
    return ca


def make_inputs(seed=0, nband=NBAND, nz=NZ, xblock=XBLOCK, xgrid=XGRID):
    """Random but realistic geometry, ancillary data and reflectances."""
    rng = np.random.default_rng(seed)
    pix = (nz, xblock, xgrid)
    band = (nband,) + pix
    u = lambda lo, hi, shape: rng.uniform(lo, hi, shape).astype(np.float32)
    return dict(
        tetas=u(20, 60, pix), tetav=u(0, 50, pix), phis=u(0, 360, pix), phiv=u(0, 360, pix),
        uh2o=u(0.5, 4, pix), uo3=u(0.25, 0.4, pix), taup550=u(0.05, 0.5, pix),
        pression=u(950, 1020, pix),
        rsurf=u(0.02, 0.5, band), k1p=u(0, 0.3, band), k2p=u(0, 0.6, band),
        # ensemble of 3 distinct aerosol models (member 0 is the reference)
        iaero=np.ascontiguousarray(np.stack([np.full(pix, m, np.int16) for m in range(NMOD)])),
    )


@pytest.fixture(scope='session')
def coeffs(smaccl_module):
    return make_coeffs(smaccl_module.type_coeff_reduced)


@pytest.fixture
def inputs():
    return make_inputs()


ANCILLARY = ('tetas', 'tetav', 'phis', 'phiv', 'uh2o', 'uo3', 'taup550', 'pression')


def run_dir(S, coeffs, inp, rsurf=None, iaero=None, **over):
    """Forward model; keyword overrides replace single inputs (e.g. uo3=...)."""
    d = dict(inp, **over)
    return S.run_dir(coeffs, *[d[k] for k in ANCILLARY],
                     d['rsurf'] if rsurf is None else rsurf, d['k1p'], d['k2p'],
                     d['iaero'] if iaero is None else iaero)


def run_inv(S, coeffs, inp, rtoa, iaero=None, **over):
    """Inverse model (atmospheric correction) of rtoa."""
    d = dict(inp, **over)
    return S.run(coeffs, *[d[k] for k in ANCILLARY], rtoa, d['k1p'], d['k2p'],
                 d['iaero'] if iaero is None else iaero)
