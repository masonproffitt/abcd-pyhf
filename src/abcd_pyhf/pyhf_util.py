import math
import multiprocessing.pool

import numpy as np

import pyhf


signal_region = 'A'
control_regions = ['B', 'C', 'D']
all_regions = [signal_region] + control_regions

poi_name = 'mu'
signal_uncertainty_name = 'systematic_uncertainty'

bkg_normalization_name = 'mu_b'
bkg_scale_factor_1_name = 'tau_B'
bkg_scale_factor_2_name = 'tau_C'


def normfactor(name):
    return {'name': name, 'type': 'normfactor', 'data': None}


bkg_normalization = normfactor(bkg_normalization_name)
bkg_modifiers = {
    signal_region: [bkg_normalization],
    control_regions[0]: [
        bkg_normalization,
        normfactor(bkg_scale_factor_1_name),
    ],
    control_regions[1]: [
        bkg_normalization,
        normfactor(bkg_scale_factor_2_name),
    ],
    control_regions[2]: [
        bkg_normalization,
        normfactor(bkg_scale_factor_1_name),
        normfactor(bkg_scale_factor_2_name),
    ],
}


def create_model(signal_yield, signal_uncertainty, blinded):
    signal_modifiers = [
        normfactor('mu'),
        {
            'name': signal_uncertainty_name,
            'type': 'normsys',
            'data': {
                'hi': 1 + signal_uncertainty,
                'lo': 1 - signal_uncertainty,
            },
        },
    ]
    regions_to_include = control_regions if blinded is True else all_regions
    spec = {
        'channels': [
            {
                'name': region,
                'samples': [
                    {
                        'name': 'signal',
                        'data': [
                            signal_yield[region] / signal_yield[signal_region]
                        ],
                        'modifiers': signal_modifiers,
                    },
                    {
                        'name': 'background',
                        'data': [1],
                        'modifiers': bkg_modifiers[region],
                    },
                ],
            }
            for region in regions_to_include
        ]
    }
    return pyhf.Model(spec)


def get_data(observed_yields, model):
    if signal_region not in observed_yields:
        data = [
            observed_yields[region] for region in control_regions
        ] + model.config.auxdata
    else:
        data = [
            observed_yields[region] for region in all_regions
        ] + model.config.auxdata
    return data


def get_par_bounds(observed_yields, model):
    if signal_region not in observed_yields:
        background_normalization_estimate = (
            observed_yields[control_regions[0]]
            * observed_yields[control_regions[1]]
            / observed_yields[control_regions[2]]
        )
        bkg_scale_factor_1_estimate = (
            observed_yields[control_regions[0]]
            / background_normalization_estimate
        )
        bkg_scale_factor_2_estimate = (
            observed_yields[control_regions[1]]
            / background_normalization_estimate
        )
    else:
        background_normalization_estimate = observed_yields[signal_region]
        bkg_scale_factor_1_estimate = (
            observed_yields[control_regions[0]]
            / observed_yields[signal_region]
        )
        bkg_scale_factor_2_estimate = (
            observed_yields[control_regions[1]]
            / observed_yields[signal_region]
        )
    background_normalization_max = (
        background_normalization_estimate
        + 5 * math.sqrt(background_normalization_estimate)
    )
    poi_max = math.ceil(background_normalization_max)
    par_bounds = model.config.suggested_bounds()
    par_bounds[model.config.par_order.index(poi_name)] = (0, poi_max)
    par_bounds[model.config.par_order.index(bkg_normalization_name)] = (
        0,
        background_normalization_max,
    )
    par_bounds[model.config.par_order.index(bkg_scale_factor_1_name)] = (
        0,
        5 * bkg_scale_factor_1_estimate,
    )
    par_bounds[model.config.par_order.index(bkg_scale_factor_2_name)] = (
        0,
        5 * bkg_scale_factor_2_estimate,
    )
    return par_bounds


def fixed_poi_fit_scan(
    data, model, init_pars, par_bounds, fixed_params, poi_values=None
):
    if poi_values is None:
        poi_max = par_bounds[model.config.par_order.index(poi_name)][1]
        poi_values = np.linspace(0, poi_max, int(poi_max) + 1)
    other_args = (data, model, init_pars, par_bounds, fixed_params)
    with multiprocessing.pool.Pool() as pool:
        results = pool.starmap(
            pyhf.infer.mle.fixed_poi_fit,
            zip(poi_values, *([arg] * len(poi_values) for arg in other_args)),
        )
    return poi_values, np.array(results)


def hypotest_scan(
    data,
    model,
    init_pars,
    par_bounds,
    fixed_params,
    poi_values=None,
    calctype='asymptotics',
    return_tail_probs=False,
    return_expected=False,
    return_expected_set=False,
):
    if poi_values is None:
        poi_max = par_bounds[model.config.par_order.index(poi_name)][1]
        poi_values = np.linspace(0, poi_max, int(poi_max) + 1)
    other_args = [
        data,
        model,
        init_pars,
        par_bounds,
        fixed_params,
        calctype,
        return_tail_probs,
        return_expected,
        return_expected_set,
    ]
    with multiprocessing.pool.Pool() as pool:
        results = pool.starmap(
            pyhf.infer.hypotest,
            zip(poi_values, *[[arg] * len(poi_values) for arg in other_args]),
        )
    cls_observed = []
    if return_tail_probs:
        tail_probs = []
    if return_expected:
        cls_expected = []
    if return_expected_set:
        cls_expected_set = []
    for result in results:
        if len(result) > 1:
            i = 0
            cls_observed.append(result[i])
            i += 1
            if return_tail_probs:
                tail_probs.append(result[i])
                i += 1
            if return_expected:
                cls_expected.append(result[i])
                i += 1
            if return_expected_set:
                cls_expected_set.append(result[i])
                i += 1
        else:
            cls_observed.append(result)
    return_values = [np.array(poi_values), np.array(cls_observed)]
    if return_tail_probs:
        return_values.append(np.array(tail_probs).T)
    if return_expected:
        return_values.append(np.array(cls_expected))
    if return_expected_set:
        return_values.append(np.array(cls_expected_set).T)
    return return_values


def poi_upper_limit(poi, cls, cl=0.95):
    return np.interp(1 - cl, cls[::-1], poi[::-1])
