#!/usr/bin/env python3

import numpy as np

from util import update_std, subdict


# Standard Parameters needed for the Deadzone HESS-EMS
STD_PARA = {
    'slope_pos': 1,
    'slope_neg': 1,
    'out_max': 1,
    'out_min': -1,
    'threshold_pos': 0.5,
    'threshold_neg': -0.5,
    'gain': 1,
    'window_up': 0.7,
    'window_low': 0.3,
    'base_max': 0.5,
    'peak_max': 0.5,
    'base_min': -0.5,
    'peak_min': -0.5
}


# Description of the standard parameters
STD_PARA_DESCRIPTIONS = {
    'slope_pos': 'Positive slope of the (saturated) deadzone function',
    'slope_neg': 'Negative slope of the (saturated) deadzone function',
    'out_max': 'Maximum output of the (saturated) deadzone function',
    'out_min': 'Minimum output of the (saturated) deadzone function, '
               'input as negative value',
    'threshold_pos': 'Positive threshold value of the (saturated) deadzone '
                     'function',
    'threshold_neg': 'Positive threshold value of the (saturated) deadzone '
                     'function, input as negative value',
    'gain': 'SOC feedback gain',
    'window_up': 'Upper window value of the feedback logic',
    'window_low': 'Lower window value of the feedback logic',
    'base_max': 'Rated positive power of base storage (charge)',
    'base_min': 'Rated negative power of base storage (discharge, input as'
                'negative value)',
    'peak_max': 'Rated positive power of peak storage (charge)',
    'peak_min': 'Rated negative power of peak storage (discharge, input as'
                'negative value)'
}


def deadzone(power_in, energy_peak, para=None):
    """
    Deadzone EMS implementation. The deadzone EMS is formulated in a way
    that depending on the chosen parameters it allows peak-prioritized
    deadzone ems, base-priortized deadzone ems, or
    base-discharge-prio-peak_charge_prio deadzone ems. It allows for
    proportional soc feedback, on/off feedback, no feedback. Further,
    the feedback reference may be defined as a constant value or as a window.
    The `power_in` is split into a base power and a peak power. How this
    dispatch is performed depends on the chosen para defined in `para`.
    For a list of para, see `STD_PARA`, for a description on these,
    see `STD_PARA_DESCRIPTIONS`. A `para` dict may be a subset of the paras
    defined in `STD_PARA`. Missing keys and values are automatically filled
    with the ones in `STD_PARA`.
    """
    # Load missing std para
    para = update_std(para, STD_PARA)

    # pass to saturated deadzone function
    dz_para = subdict(
        para,
        ['slope_pos', 'slope_neg', 'out_max', 'out_min',
         'threshold_pos', 'threshold_neg']
    )
    peak_pre = _sat_deadzone(power_in, dz_para)
    base_pre = power_in - peak_pre

    # pass to soc control - feedback calc, reserve calc, saturation
    fb_para = subdict(para, ['gain', 'window_up', 'window_low'])
    feedback_pre = _feedback(energy_peak, fb_para)

    res_para = subdict(para, ['base_max', 'base_min', 'peak_max', 'peak_min'])
    sat_low, sat_high = _reserve(base_pre, peak_pre, res_para)

    sat_para = {'sat_low': sat_low, 'sat_high': sat_high}
    feedback = _saturation(feedback_pre, sat_para)

    # add feedback and deadzone calculation
    peak = peak_pre + feedback
    base = base_pre - feedback

    return base, peak


def _sat_deadzone(val_in, para):
    """
    Saturated deadzone function: The `val_in` is piped into a classic
    deadzone function and then into a classic saturation function
    Para dict contains:
    ['slope_pos', 'slope_neg', 'out_max', 'out_min',
    'threshold_pos', 'threshold_neg']
    """
    # Extract paras from dict
    dz_para = subdict(para, ['slope_pos', 'slope_neg',
                             'threshold_pos', 'threshold_neg'])
    dz = _deadzone(val_in, dz_para)

    sat_para = {'sat_low': para['out_min'], 'sat_high': para['out_max']}
    dz_sat = _saturation(dz, sat_para)
    return dz_sat


def _feedback(val_in, para):
    """
    Computes soc feedback based on the current soc `val_in` and the gain and
    reference window stored in `para`. The implementation is a proportional
    feedback where the feedback output val linearily rises with the
    deviation from the set point. The set point is not a constant,
    but a window with an upper and a lower value. If `val_in` is within this
    window, the output will be zero.
    Para dict contains ['gain', 'window_up', 'window_low']
    """
    k = para['gain']
    up = para['window_up']
    low = para['window_low']

    diff = ((val_in - low) * (val_in <= low) +
            0              * (low < val_in < up) +  # noqa
            (val_in - up)  * (val_in >= up))  # noqa
    return k*diff


def _reserve(base_in, peak_in, para):
    """
    Based on input base and peak storage power `base_in` and `peak_in` and
    given bounds for these storages defined in `para`, the function
    calculates an upper and lower bounds of how much an soc feedback signal
    may alter the input powers
    Para dict contains ['base_max', 'base_min', 'peak_max', 'peak_min']
    """
    base_max = para['base_max']
    base_min = para['base_min']
    peak_max = para['peak_max']
    peak_min = para['peak_min']

    high = np.minimum(peak_max - peak_in, base_in - base_min)
    low = np.minimum(peak_min - peak_in, base_in - base_max)
    return low, high


def _saturation(s_in, para):
    """
    Classic saturation function. If the input value `s_in` falls below a
    limit or exceeds a limit, the limit is returned, else, the input value
    is returned.
    Para dict contains: ['sat_low', 'sat_high']
    """
    pos = para['sat_low']
    neg = para['sat_high']

    s_out = (pos * (s_in >= pos) +
             s_in * (neg < s_in < pos) +
             neg * (s_in <= neg))
    return s_out


def _deadzone(d_in, para):
    """
    Classic deadzone function. If the input `d_in` is within an upper or
    lower threshold value, the output is zero. Above and below this value,
    the output rises and falls linearily to the difference of the limt.
    Para dict contains:
    ['slope_pos', 'slope_neg', 'threshold_pos', 'threshold_neg']
    """
    mp = para['slope_neg']
    mn = para['slope_pos']
    tp = para['threshold_pos']
    tn = para['threshold_neg']

    d_out = (mp*(d_in - tp) * (d_in > tp) +
             0              * (tn <= d_in <= tp) +  # noqa
             mn*(d_in - tn) * (d_in < tn))
    return d_out
