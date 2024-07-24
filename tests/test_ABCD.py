import math

import pyhf

from abcd_pyhf import ABCD


observed_yields = {'A': 10, 'B': 20, 'C': 30, 'D': 60}

signal_yields = {'A': 35, 'B': 7, 'C': 5, 'D': 1}

signal_uncertainty = 0.1

background_uncertainty = 0.05


def test_init():
    assert ABCD(observed_yields, signal_yields, signal_uncertainty) is not None


def test_observed_yields():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.observed_yields == observed_yields


def test_signal_yields():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.signal_yields == signal_yields


def test_signal_uncertainty():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.signal_uncertainty == signal_uncertainty


def test_background_uncertainty():
    abcd = ABCD(
        observed_yields,
        signal_yields,
        signal_uncertainty,
        background_uncertainty,
    )
    assert abcd.background_uncertainty == background_uncertainty


def test_blinded():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.blinded is False
    blinded_yields = observed_yields.copy()
    del blinded_yields['A']
    abcd_blinded = ABCD(blinded_yields, signal_yields, signal_uncertainty)
    assert abcd_blinded.blinded is True


def test_model():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert isinstance(abcd.model, pyhf.Model)


def test_data():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.data is not None


def test_init_pars():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.init_pars is not None


def test_par_bounds():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.par_bounds is not None


def test_fixed_params():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.fixed_params() is not None
    assert abcd.fixed_params(bkg_only=False) is not None
    assert abcd.fixed_params(bkg_only=True) is not None
    assert abcd.fixed_params(False) is not None
    assert abcd.fixed_params(True) is not None


def test_fixed_poi_fit():
    observed_yields_copy = observed_yields.copy()
    observed_yields_copy['A'] += signal_yields['A']
    observed_yields_copy['B'] += signal_yields['B']
    observed_yields_copy['C'] += signal_yields['C']
    observed_yields_copy['D'] += signal_yields['D']
    abcd = ABCD(observed_yields_copy, signal_yields, signal_uncertainty)
    fixed_poi_fit = abcd._fixed_poi_fit(signal_yields['A'])
    assert (
        fixed_poi_fit[abcd.model.config.par_names.index('mu')][0]
        == signal_yields['A']
    )
    assert math.isclose(
        fixed_poi_fit[abcd.model.config.par_names.index('signal_uncertainty')][
            0
        ],
        0,
        abs_tol=1e-1,
    )
    assert math.isclose(
        fixed_poi_fit[abcd.model.config.par_names.index('mu_b')][0],
        observed_yields['A'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        fixed_poi_fit[abcd.model.config.par_names.index('tau_B')][0],
        observed_yields['B'] / observed_yields['A'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        fixed_poi_fit[abcd.model.config.par_names.index('tau_C')][0],
        observed_yields['C'] / observed_yields['A'],
        rel_tol=1e-2,
    )


def test_bkg_only_fit():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    bkg_only_fit = abcd.bkg_only_fit()
    assert bkg_only_fit[abcd.model.config.par_names.index('mu')][0] == 0
    assert (
        bkg_only_fit[abcd.model.config.par_names.index('signal_uncertainty')][
            0
        ]
        == 0
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('mu_b')][0],
        observed_yields['A'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('tau_B')][0],
        observed_yields['B'] / observed_yields['A'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('tau_C')][0],
        observed_yields['C'] / observed_yields['A'],
        rel_tol=1e-2,
    )


def test_fit():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    fit = abcd.fit()
    assert math.isclose(
        fit[abcd.model.config.par_names.index('mu')][0], 0, abs_tol=1e-1
    )
    assert math.isclose(
        fit[abcd.model.config.par_names.index('signal_uncertainty')][0],
        0,
        abs_tol=1e-1,
    )
    assert math.isclose(
        fit[abcd.model.config.par_names.index('mu_b')][0],
        observed_yields['A'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        fit[abcd.model.config.par_names.index('tau_B')][0],
        observed_yields['B'] / observed_yields['A'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        fit[abcd.model.config.par_names.index('tau_C')][0],
        observed_yields['C'] / observed_yields['A'],
        rel_tol=1e-2,
    )


def test_twice_nll_scan():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd._twice_nll_scan()


def test_twice_nll():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.twice_nll() is not None


def test_twice_nll_plot():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.twice_nll_plot() is not None


def test_hypotest():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd._hypotest(1)


def test_hypotest_scan():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd._hypotest_scan()


def test_clsb():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.clsb() is not None


def test_clb():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.clb() is not None


def test_cls():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.cls() is not None


def test_upper_limit():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    observed_upper_limit, expected_upper_limit = abcd.upper_limit()
    assert math.isclose(
        observed_upper_limit, expected_upper_limit[2], rel_tol=1e-2
    )


def test_brazil_plot():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    assert abcd.brazil_plot() is not None


# https://github.com/masonproffitt/abcd-pyhf/issues/14
def test_zero_events():
    observed_yields_copy = observed_yields.copy()
    observed_yields_copy['A'] = 0
    abcd = ABCD(observed_yields_copy, signal_yields, signal_uncertainty)
    assert abcd.par_bounds is not None


# https://github.com/masonproffitt/abcd-pyhf/issues/20
def test_bkg_only_fit_very_small_expected_mu_b():
    observed_yields_copy = observed_yields.copy()
    observed_yields_copy['A'] = 0
    observed_yields_copy['D'] *= 1000
    abcd = ABCD(observed_yields_copy, signal_yields, signal_uncertainty)
    bkg_only_fit = abcd.bkg_only_fit()
    assert bkg_only_fit[abcd.model.config.par_names.index('mu')][0] == 0
    assert (
        bkg_only_fit[abcd.model.config.par_names.index('signal_uncertainty')][
            0
        ]
        == 0
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('mu_b')][0],
        observed_yields_copy['A'],
        abs_tol=1e-1,
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('tau_B')][0],
        observed_yields_copy['D'] / observed_yields_copy['C'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('tau_C')][0],
        observed_yields_copy['D'] / observed_yields_copy['B'],
        rel_tol=1e-2,
    )


# https://github.com/masonproffitt/abcd-pyhf/issues/22
def test_bkg_only_fit_special_case():
    observed_yields_special_case = {
        'A': 0,
        'B': 15004,
        'C': 441,
        'D': 192036934,
    }
    signal_yields_special_case = {
        'A': 0.13,
        'B': 0.004,
        'C': 0.00001,
        'D': 0.00006,
    }
    signal_uncertainty_special_case = 0.02
    abcd = ABCD(
        observed_yields_special_case,
        signal_yields_special_case,
        signal_uncertainty_special_case,
    )
    bkg_only_fit = abcd.bkg_only_fit()
    assert bkg_only_fit[abcd.model.config.par_names.index('mu')][0] == 0
    assert (
        bkg_only_fit[abcd.model.config.par_names.index('signal_uncertainty')][
            0
        ]
        == 0
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('mu_b')][0],
        observed_yields_special_case['A'],
        abs_tol=1e-1,
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('tau_B')][0],
        observed_yields_special_case['D'] / observed_yields_special_case['C'],
        rel_tol=1e-2,
    )
    assert math.isclose(
        bkg_only_fit[abcd.model.config.par_names.index('tau_C')][0],
        observed_yields_special_case['D'] / observed_yields_special_case['B'],
        rel_tol=1e-2,
    )


# https://github.com/masonproffitt/abcd-pyhf/issues/25
def test_fixed_poi_fit_special_case():
    observed_yields_special_case = {
        'A': 0,
        'B': 15004,
        'C': 441,
        'D': 192036934,
    }
    signal_yields_special_case = {
        'A': 0.13,
        'B': 0.004,
        'C': 0.00001,
        'D': 0.00006,
    }
    signal_uncertainty_special_case = 0.02
    abcd = ABCD(
        observed_yields_special_case,
        signal_yields_special_case,
        signal_uncertainty_special_case,
    )
    assert abcd._fixed_poi_fit(4) is not None


def test_toys():
    observed_yields_low = observed_yields.copy()
    for key in observed_yields:
        observed_yields_low[key] //= 10
    abcd = ABCD(observed_yields_low, signal_yields, signal_uncertainty)
    abcd._hypotest_scan(calctype='toybased', ntoys=10)


def test_hypotest_special_case():
    observed_yields_special_case = {
        'A': 0,
        'B': 15004,
        'C': 441,
        'D': 192036934,
    }
    signal_yields_special_case = {
        'A': 0.13,
        'B': 0.004,
        'C': 0.00001,
        'D': 0.00006,
    }
    signal_uncertainty_special_case = 0.02
    abcd = ABCD(
        observed_yields_special_case,
        signal_yields_special_case,
        signal_uncertainty_special_case,
    )
    abcd._hypotest(0)
