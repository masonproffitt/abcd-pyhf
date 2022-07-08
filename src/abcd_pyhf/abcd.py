import functools

import numpy as np
import matplotlib.pyplot as plt

import pyhf
import pyhf.contrib.viz.brazil

from .pyhf_util import (
    signal_region,
    poi_name,
    bkg_normalization_name,
    create_model,
    get_data,
    get_init_pars,
    get_par_bounds,
    get_fixed_params,
    fixed_poi_fit,
    fit,
    fixed_poi_fit_scan,
    hypotest_scan,
    poi_upper_limit,
)


pyhf.set_backend(pyhf.default_backend, pyhf.optimize.scipy_optimizer(solver_options={'eps': 1e-7}))


class ABCD:
    def __init__(
        self, observed_yields, signal_yields=None, signal_uncertainty=None
    ):
        self._observed_yields = observed_yields
        self._signal_yields = signal_yields
        self._signal_uncertainty = signal_uncertainty

    @property
    def observed_yields(self):
        return self._observed_yields

    @property
    def signal_yields(self):
        return self._signal_yields

    @property
    def signal_uncertainty(self):
        return self._signal_uncertainty

    @property
    def blinded(self):
        return signal_region not in self.observed_yields

    @property
    def model(self):
        return create_model(
            self.signal_yields, self.signal_uncertainty, self.blinded
        )

    @property
    def data(self):
        return get_data(self.observed_yields, self.model)

    @property
    def init_pars(self):
        return get_init_pars(self.observed_yields, self.model)

    @property
    def par_bounds(self):
        return get_par_bounds(self.observed_yields, self.model)

    def fixed_params(self, bkg_only=False):
        return get_fixed_params(self.model, bkg_only=bkg_only)

    def _fixed_poi_fit(self, poi_value, return_uncertainties=True):
        pars = fixed_poi_fit(
            poi_value,
            data=self.data,
            pdf=self.model,
            init_pars=self.init_pars,
            par_bounds=self.par_bounds,
            fixed_params=self.fixed_params(),
            return_uncertainties=return_uncertainties,
        )
        return pars

    def bkg_only_fit(self, return_uncertainties=True):
        pars = fixed_poi_fit(
            poi_val=0,
            data=self.data,
            pdf=self.model,
            init_pars=self.init_pars,
            par_bounds=self.par_bounds,
            fixed_params=self.fixed_params(bkg_only=True),
            return_uncertainties=return_uncertainties,
        )
        return pars

    def fit(self, return_uncertainties=True):
        pars = fit(
            data=self.data,
            pdf=self.model,
            init_pars=self.init_pars,
            par_bounds=self.par_bounds,
            fixed_params=self.fixed_params(),
            return_uncertainties=return_uncertainties,
        )
        return pars

    def bkg_only_signal_region_estimate(self):
        return tuple(
            self.bkg_only_fit()[
                self.model.config.par_order.index(bkg_normalization_name)
            ]
        )

    def _twice_nll_scan(self):
        if not hasattr(self, '_twice_nll_scan_result'):
            poi_values, pars_set = fixed_poi_fit_scan(
                self.data,
                self.model,
                self.init_pars,
                self.par_bounds,
                self.fixed_params(),
            )
            best_fit_pars = np.array(self.fit(return_uncertainties=False))
            best_fit_twice_nll = pyhf.infer.mle.twice_nll(
                pars=best_fit_pars, data=self.data, pdf=self.model
            )
            setattr(
                self,
                '_twice_nll_scan_result',
                (
                    poi_values,
                    np.array(
                        [
                            pyhf.infer.mle.twice_nll(
                                pars, self.data, self.model
                            )
                            - best_fit_twice_nll
                            for pars in pars_set
                        ]
                    ),
                ),
            )
        return getattr(self, '_twice_nll_scan_result')

    def twice_nll(self):
        return self._twice_nll_scan()

    def twice_nll_plot(self):
        plt.xlabel(r'$\mu$')
        plt.xlim(
            0,
            self.par_bounds[self.model.config.par_names().index(poi_name)][1],
        )
        plt.ylabel(r'$-2 \ln L$')
        plt.ylim(0, 5)
        poi_values, twice_nll_values = self.twice_nll()
        return plt.plot(poi_values, twice_nll_values)[0]

    def _hypotest(self, poi_value, calctype='asymptotics', **kwargs):
        return pyhf.infer.hypotest(
            poi_value,
            self.data,
            self.model,
            self.init_pars,
            self.par_bounds,
            self.fixed_params(),
            calctype=calctype,
            return_tail_probs=True,
            return_expected_set=True,
            **kwargs
        )

    @functools.lru_cache()
    def _hypotest_scan(self, calctype='asymptotics', **kwargs):
        return hypotest_scan(
            self.data,
            self.model,
            self.init_pars,
            self.par_bounds,
            self.fixed_params(),
            calctype=calctype,
            return_tail_probs=True,
            return_expected_set=True,
            **kwargs
        )

    def clsb(self, calctype='asymptotics', **kwargs):
        return self._hypotest_scan(calctype=calctype, **kwargs)[0], self._hypotest_scan(calctype=calctype, **kwargs)[2][0]

    def clb(self, calctype='asymptotics', **kwargs):
        return self._hypotest_scan(calctype=calctype, **kwargs)[0], self._hypotest_scan(calctype=calctype, **kwargs)[2][1]

    def cls(self, calctype='asymptotics', **kwargs):
        return (
            self._hypotest_scan(calctype=calctype, **kwargs)[0],
            self._hypotest_scan(calctype=calctype, **kwargs)[1],
            self._hypotest_scan(calctype=calctype, **kwargs)[3],
        )

    def upper_limit(self, cl=0.95, calctype='asymptotics', **kwargs):
        poi, cls_observed, cls_expected_set = self.cls(calctype=calctype, **kwargs)
        return poi_upper_limit(poi, cls_observed), [
            poi_upper_limit(poi, cls_expected)
            for cls_expected in cls_expected_set
        ]

    def brazil_plot(self, calctype='asymptotics', **kwargs):
        mu, cls_observed, cls_expected_set = self.cls(calctype=calctype, **kwargs)
        results = list(zip(cls_observed, cls_expected_set.T))
        pyhf.contrib.viz.brazil.plot_results(mu, results)
