from abcd_pyhf import ABCD


observed_yields = {'A': 1,
                   'B': 2,
                   'C': 3,
                   'D': 4}

signal_yields = {'A': 1,
                 'B': 2,
                 'C': 3,
                 'D': 4}

signal_uncertainty = 0.1


def test_init():
    ABCD(observed_yields, signal_yields, signal_uncertainty)


def test_observed_yields():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.observed_yields


def test_signal_yields():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.signal_yields


def test_signal_uncertainty():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.signal_uncertainty


def test_blinded():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.blinded


def test_model():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.model


def test_data():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.data


def test_init_pars():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.init_pars


def test_par_bounds():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.par_bounds


def test_fixed_params():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.fixed_params


def test_bkg_only_fit():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.bkg_only_fit()


def test_fit():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.fit()


def test_bkg_only_signal_region_estimate():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.bkg_only_signal_region_estimate()


def test_twice_nll_scan():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd._twice_nll_scan()


def test_twice_nll():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.twice_nll()


def test_twice_nll_plot():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.twice_nll_plot()


def test_hypotest_scan():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd._hypotest_scan()


def test_clsb():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.clsb()


def test_clb():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.clb()


def test_cls():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.cls()


def test_upper_limit():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.upper_limit()


def test_brazil_plot():
    abcd = ABCD(observed_yields, signal_yields, signal_uncertainty)
    abcd.brazil_plot()
