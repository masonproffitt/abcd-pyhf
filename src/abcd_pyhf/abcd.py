import numpy as np
import matplotlib.pyplot as plt

import pyhf
import pyhf.contrib.viz.brazil

from .pyhf_util import (
    signal_region,
    poi_name,
    create_model,
    get_data,
    get_init_pars,
    get_par_bounds,
    get_fixed_params,
    fixed_poi_fit_scan,
    hypotest_scan,
    poi_upper_limit,
)


pyhf.set_backend('numpy', 'minuit')


class ABCD:
    """
    An ABCD plane, including yields in data and signal and systematic
    uncertainties

    Parameters
    ----------
    observed_yields : dict
        Number of events observed in each region, in {region_name: n_events}
        format. The control region yields must be specified, but the signal
        region yield is optional.
    signal_yields : dict, optional
        Signal event yield in each region, in {region_name: n_events} format.
        Only the relative proportions matter; the overall normalization is
        arbitrary.
    signal_uncertainty : float, optional
        Total fractional uncertainty of the signal yields

    Attributes
    ----------
    observed_yields : dict
        Number of events observed in each region, in {region_name: n_events}
        format
    signal_yields : dict or None
        Signal event yield in each region, in {region_name: n_events} format.
        Only the relative proportions matter; the overall normalization is
        arbitrary.
    signal_uncertainty : float or None
        Total fractional uncertainty of the signal yields
    blinded : bool
        Whether the signal region is blinded or not. False if the observed
        number of events in the signal region was provided, otherwise True.
    model : pyhf.pdf.Model
        pyhf model of this ABCD plane
    data : list
        pyhf data for this ABCD plane
    init_pars : list
        Initial parameter values
    par_bounds : list
        Parameter bounds
    """

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
        """
        Identify which parameters are fixed

        Parameters
        ----------
        bkg_only : bool, optional
            Whether the signal strength should be fixed to zero or not

        Returns
        -------
        list
            A boolean value for each parameter: True if the parameter is fixed,
            otherwise False
        """
        return get_fixed_params(self.model, bkg_only=bkg_only)

    def _fixed_poi_fit(self, poi_value):
        pars = pyhf.infer.mle.fixed_poi_fit(
            poi_value,
            data=self.data,
            pdf=self.model,
            init_pars=self.init_pars,
            par_bounds=self.par_bounds,
            fixed_params=self.fixed_params(),
            return_uncertainties=True,
        )
        return pars

    def bkg_only_fit(self):
        """
        Perform a background-only fit to the observed yields

        Returns
        -------
        numpy.ndarray
            Fit value and its uncertainty for each parameter
        """
        pars = pyhf.infer.mle.fixed_poi_fit(
            poi_val=0,
            data=self.data,
            pdf=self.model,
            init_pars=self.init_pars,
            par_bounds=self.par_bounds,
            fixed_params=self.fixed_params(bkg_only=True),
            return_uncertainties=True,
        )
        return pars

    def fit(self):
        """
        Perform a fit to the observed yields

        Returns
        -------
        numpy.ndarray
            Fit value and its uncertainty for each parameter
        """
        pars = pyhf.infer.mle.fit(
            data=self.data,
            pdf=self.model,
            init_pars=self.init_pars,
            par_bounds=self.par_bounds,
            fixed_params=self.fixed_params(),
            return_uncertainties=True,
        )
        return pars

    def _twice_nll_scan(self):
        if not hasattr(self, '_twice_nll_scan_result'):
            poi_values, pars_set = fixed_poi_fit_scan(
                self.data,
                self.model,
                self.init_pars,
                self.par_bounds,
                self.fixed_params(),
            )
            best_fit_pars = np.array(self.fit()).T[0]
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
        """
        Calculate the negative log-likelihood times two for a variety of signal
        strengths

        Returns
        -------
        tuple
            Mu values and the corresponding negative log-likelihoods times two
            for each signal strength
        """
        return self._twice_nll_scan()

    def twice_nll_plot(self):
        """
        Plot the negative log-likelihood times two versus signal strength

        Returns
        -------
        matplotlib.lines.Line2D
            Line representing the plotted data
        """
        plt.xlabel(r'$\mu$')
        plt.xlim(
            0,
            self.par_bounds[self.model.config.par_names().index(poi_name)][1],
        )
        plt.ylabel(r'$-2 \ln L$')
        plt.ylim(0, 5)
        poi_values, twice_nll_values = self.twice_nll()
        return plt.plot(poi_values, twice_nll_values)[0]

    def _hypotest_scan(self):
        if not hasattr(self, '_hypotest_scan_result'):
            setattr(
                self,
                '_hypotest_scan_result',
                hypotest_scan(
                    self.data,
                    self.model,
                    self.init_pars,
                    self.par_bounds,
                    self.fixed_params(),
                    return_tail_probs=True,
                    return_expected_set=True,
                ),
            )
        return getattr(self, '_hypotest_scan_result')

    def clsb(self):
        """
        Calculate CL_{s+b} for a variety of signal strengths

        Returns
        -------
        tuple
            Mu values and the corresponding CL_{s+b} values for each signal
            strength
        """
        return self._hypotest_scan()[0], self._hypotest_scan()[2][0]

    def clb(self):
        """
        Calculate CL_b for a variety of signal strengths

        Returns
        -------
        tuple
            Mu values and the corresponding CL_b values for each signal
            strength
        """
        return self._hypotest_scan()[0], self._hypotest_scan()[2][1]

    def cls(self):
        """
        Calculate CL_s for a variety of signal strengths

        Returns
        -------
        tuple
            Mu values and the corresponding CL_s values for each signal
            strength
        """
        return (
            self._hypotest_scan()[0],
            self._hypotest_scan()[1],
            self._hypotest_scan()[3],
        )

    def upper_limit(self, cl=0.95):
        """
        Calculate the upper limit on the signal strength

        Parameters
        ----------
        cl : float, optional
            Conflidence level of the upper limit

        Returns
        -------
        tuple
            Observed upper limit on mu and the expected upper limit band in the
            form: (-2 sigma, -1 sigma, median, +1 sigma, +2 sigma)
        """
        poi, cls_observed, cls_expected_set = self.cls()
        return poi_upper_limit(poi, cls_observed), [
            poi_upper_limit(poi, cls_expected)
            for cls_expected in cls_expected_set
        ]

    def brazil_plot(self):
        """
        Make a Brazil plot of CL_s and its components versus signal strength

        Returns
        -------
        pyhf.contrib.viz.brazil.BrazilBandCollection
            Artist containing the matplotlib.artist objects drawn
        """
        mu, cls_observed, cls_expected_set = self.cls()
        results = list(zip(cls_observed, cls_expected_set.T))
        return pyhf.contrib.viz.brazil.plot_results(mu, results)
