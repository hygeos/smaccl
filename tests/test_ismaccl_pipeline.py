"""ISmaccl pipeline (exec_smaccl) in both kernel modes."""
import numpy as np
import pytest

NPIX = 50   # not a multiple of XBLOCK * XGRID: exercises the NaN padding
XBLOCK, XGRID = 4, 8

CONFIG = dict(nmodels=3, kp_map_file=None, legend_file=None, mapping_file=None,
              epre=1., etaup=0.05, ertaup=0.1, euo3=0.01, eruo3=0.05, euh2o=0.1, eruh2o=0.1)


@pytest.fixture(scope='module')
def ismaccl(smaccl_module):
    for mod in ('xarray', 'dask', 'h5py', 'scipy', 'pandas', 'PIL'):
        pytest.importorskip(mod)
    from smaccl import ISmaccl
    return ISmaccl


@pytest.fixture(autouse=True)
def no_koppen_geiger(ismaccl, monkeypatch):
    """The Koppen-Geiger map files are not shipped; they only scale Dtaup."""
    monkeypatch.setattr(ismaccl, 'KoppenGeiger', lambda **kw: None)
    monkeypatch.setattr(ismaccl, 'get_aod_uncertainty_multiplier', lambda lat, lon, kg: np.ones_like(lat))


@pytest.fixture(scope='module')
def pixels():
    rng = np.random.default_rng(1)
    u = lambda lo, hi, s=NPIX: rng.uniform(lo, hi, s).astype(np.float32)
    nband = 2
    return dict(
        geo=dict(sza=u(20, 60), vza=u(0, 50), saa=u(0, 360), vaa=u(0, 360), taup550=u(0.05, 0.5),
                 uh2o=u(0.5, 4), uo3=u(0.25, 0.4), p0=u(980, 1020), t10m=u(270, 300),
                 alt=u(0, 500), Dalt=u(0, 10), lat=u(-60, 60), lon=u(-180, 180)),
        rsurf=u(0.02, 0.5, (nband, NPIX)),
        band_err=np.full((nband, NPIX), 0.01, np.float32),
        k1p=u(0, 0.3, (nband, NPIX)), k2p=u(0, 0.6, (nband, NPIX)),
        iaero=np.stack([np.full(NPIX, m) for m in range(3)]).astype(np.int16),
    )


def exec_smaccl(S, coeffs, px, band):
    g = {k: v.copy() for k, v in px['geo'].items()}   # exec_smaccl edits lat/lon in place
    return S.exec_smaccl(g['sza'], g['vza'], g['saa'], g['vaa'], g['taup550'], g['uh2o'], g['uo3'],
                         g['p0'], g['t10m'], g['alt'], g['Dalt'], g['lat'], g['lon'], band,
                         px['band_err'], None, None, coeffs, px['iaero'], px['k1p'], px['k2p'])


@pytest.fixture(scope='module')
def pipelines(ismaccl):
    from conftest import make_coeffs
    from smaccl.smaccl import type_coeff_reduced
    ca = make_coeffs(type_coeff_reduced)
    inv = ismaccl.ISmaccl(CONFIG, None, ca, XBLOCK=XBLOCK, XGRID=XGRID, platform='CPU')
    fwd = ismaccl.ISmaccl(CONFIG, None, ca, XBLOCK=XBLOCK, XGRID=XGRID, platform='CPU', mode='smaccl_dir')
    return ca, inv, fwd


def test_mode_default_and_forwarded(pipelines):
    _, inv, fwd = pipelines
    assert inv.mode == 'smaccl' and inv.smaccl.mode == 'smaccl'
    assert fwd.mode == 'smaccl_dir' and fwd.smaccl.mode == 'smaccl_dir'


def test_invalid_mode_rejected(ismaccl):
    with pytest.raises(ValueError):
        ismaccl.ISmaccl(CONFIG, None, None, platform='CPU', mode='both')


def test_pipeline_round_trip(pipelines, pixels):
    ca, inv, fwd = pipelines
    out_f = exec_smaccl(fwd, ca, pixels, pixels['rsurf'])
    assert all(np.isfinite(a).all() for a in out_f[:8])
    assert out_f[0].shape == pixels['rsurf'].shape
    out_i = exec_smaccl(inv, ca, pixels, out_f[0])
    np.testing.assert_allclose(out_i[0], pixels['rsurf'], rtol=1e-5)
    # output order: rsurf, rsurf0, dev_std, Juh2o, Juo3, Jrtoa, ...
    np.testing.assert_allclose(out_f[5] * out_i[5], 1., rtol=1e-4)


def test_pipeline_matches_direct_smaccl_call(pipelines, pixels, smaccl_module):
    """exec_smaccl padding / reshaping does not alter the kernel results."""
    from smaccl.c3s_lib import Ps
    ca, _, fwd = pipelines
    g = pixels['geo']
    out = exec_smaccl(fwd, ca, pixels, pixels['rsurf'])
    S = fwd.smaccl
    sh = (1, 1, NPIX)
    r = lambda a: np.ascontiguousarray(a.reshape(sh), np.float32)
    rb = lambda a: np.ascontiguousarray(a.reshape((a.shape[0],) + sh), np.float32)
    ref = S.run_dir(ca, r(g['sza']), r(g['vza']), r(g['saa']), r(g['vaa']), r(g['uh2o']), r(g['uo3']),
                    r(g['taup550']), r(Ps(g['alt'], g['p0'], g['t10m'])), rb(pixels['rsurf']),
                    rb(pixels['k1p']), rb(pixels['k2p']),
                    np.ascontiguousarray(pixels['iaero'].reshape((3,) + sh)))
    np.testing.assert_allclose(out[0], ref[0].reshape(out[0].shape), rtol=1e-6)
