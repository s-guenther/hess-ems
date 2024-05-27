#!/usr/bin/env python3
"""
Deadzone-based EMS Module

This module implements a Deadzone Energy Management Strategy (EMS) for hybrid
energy storage systems. It offers various operational modes for peak and base
power management with adjustable parameters for system optimization.


Main method/EMS implementation:

deadzone(power_in, energy_peak, para=None)
    Implements a versatile Deadzone EMS supporting multiple operational modes,
    including peak/base prioritization and feedback control. It intelligently
    splits input power and adjusts it based on the provided or parameters.

Details on the parameterization of this EMS can be found in the variables
    `STD_PARA` and
    `STD_PARA_DESCRIPTIONS`.

Internal functions:
    _deadzone
    _saturation
    _reserve
    _feedback
    _sat_deadzone
"""

import numpy as np

from hessems.util import update_std, subdict


# Standard Parameters needed for the Deadzone HESS-EMS
STD_PARA = {
    'slope_pos': 1,
    'slope_neg': 1,
    'out_max': 0.5,
    'out_min': -0.5,
    'threshold_pos': 0.5,
    'threshold_neg': -0.5,
    'gain': 1e-1,
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
    Deadzone Energy Management Strategy (EMS).

    Depending on chosen parameterization, this Deadzone EMS offers various 
    operational modes including peak-prioritized, base-prioritized, and a
    combination of base-discharge-priority and peak-charge-priority. It
    supports proportional SoC feedback, on/off feedback, or no feedback, with
    a feedback reference defined as either a constant value or a window. The
    `power_in` is split into base and peak powers, with the dispatch method
    determined by the provided `para` dictionary. If `para` is not provided or
    incomplete, defaults from `STD_PARA` are used. Information on these 
    standard parameters can be found in `STD_PARA_DESCRIPTIONS`.

    Parameters
    ----------
    power_in : scalar or arraylike (numpy)
        Input power to be managed by the EMS.
    energy_peak : scalar or arraylike (numpy)
        Peak energy input for feedback calculation.
    para : dict, optional
        Parameters for EMS operation. Defaults are filled from `STD_PARA`.
        A description of the possible keys can be found in 
        `STD_PARA_DESCRIPTIONS`.

    Returns
    -------
    base : float or numpy.array
        Power dispatched to base storage.
    peak : float or numpy.array
        Power dispatched to peak storage.
    base_pre : floa or numpy.array
        Internal base power before energy feedback adjustment
    peak_pre : floa or numpy.array
        Internal peak power before energy feedback adjustment
    feedback : floa or numpy.array
        Dynamically saturated feedback that is added to the *_pre outputs
    feedback_pre : floa or numpy.array
        Unsaturated feedback calculated from the feedback logic
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

    return base, peak, base_pre, peak_pre, feedback, feedback_pre


def _sat_deadzone(val_in, para):
    """
    Apply a saturated deadzone operation to an input value.

    This function processes the input `val_in` through a deadzone function
    followed by a saturation function. The deadzone function filters the
    input within a specified threshold range, and the saturation function
    limits the output to defined maximum and minimum values.

    Parameters
    ----------
    val_in : scalar or arraylike (numpy)
        Input value to be processed.
    para : dict
        A dictionary containing parameters for the saturated deadzone
        operation. Includes 'slope_pos', 'slope_neg', 'out_max', 'out_min',
        'threshold_pos', and 'threshold_neg'.

    Returns
    -------
    ndarray
        The processed value after applying saturated deadzone operation.
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
    Compute state of charge (SoC) feedback.

    This function calculates the SoC feedback based on the current state of
    charge (`val_in`), and parameters 'gain', 'window_up', and 'window_low'
    in `para`. It implements a proportional feedback where the output
    linearly varies with the deviation from a reference window. If `val_in`
    is within this window, the output is zero.

    Parameters
    ----------
    val_in : scalar or arraylike (numpy)
        The current state of charge value.
    para : dict
        Parameter dictionary containing 'gain', 'window_up', and 'window_low'.

    Returns
    -------
    scalar or ndarray
        The feedback value, based on the state of charge deviation from the
        reference window.
    """
    k = para['gain']
    up = para['window_up']
    low = para['window_low']

    diff = ((val_in - low) * (val_in <= low) +
            0              * ((low < val_in) & (val_in < up)) +  # noqa
            (val_in - up)  * (val_in >= up))  # noqa
    return -k*diff


def _reserve(base_in, peak_in, para):
    """
    Calculate upper and lower bounds for power adjustments.

    This function computes the permissible range for adjusting the base and
    peak storage powers (`base_in` and `peak_in`). It ensures these
    adjustments stay within specified bounds defined in the `para` dictionary.

    Parameters
    ----------
    base_in : scalar or arraylike (numpy)
        Base storage power input.
    peak_in : scalar or arraylike (numpy)
        Peak storage power input.
    para : dict
        Parameter dictionary containing 'base_max', 'base_min', 'peak_max',
        and 'peak_min', which define the operational limits for base and peak
        storages.

    Returns
    -------
    tuple
        A tuple containing the lower and upper bounds for the state of charge
        (SoC) feedback signal's impact on the input powers.
    """
    base_max = para['base_max']
    base_min = para['base_min']
    peak_max = para['peak_max']
    peak_min = para['peak_min']

    low = np.maximum(peak_min - peak_in, base_in - base_max)
    high = np.minimum(peak_max - peak_in, base_in - base_min)
    return low, high


def _saturation(s_in, para):
    """
    Classic saturation function.

    This function applies a saturation operation to a signal. Values that
    exceed specified upper and lower limits are capped at these limits, while
    values within the limits remain unchanged.

    Parameters
    ----------
    s_in : scalar or arraylike (numpy)
        Input values to be processed by the saturation function.
    para : dict
        Parameter dictionary containing the following keys:
        'sat_low' and 'sat_high', defining the lower and upper saturation
        limits.

    Returns
    -------
    ndarray
        The processed signal after applying the saturation operation, with
        values exceeding the limits capped at 'sat_high' or 'sat_low'.
    """
    neg = para['sat_low']
    pos = para['sat_high']

    s_out = (pos * (s_in >= pos) +
             s_in * ((neg < s_in) & (s_in < pos)) +
             neg * (s_in <= neg))
    return s_out


def _deadzone(d_in, para):
    """
    Classic deadzone function.

    This function applies a deadzone operation to a signal. Values within a
    specified range, defined by by an upper and lower threshold are set to
    zero. Values above the threshold linearily rise and fall with a
    specified slope.

    Parameters
    ----------
    d_in : scalar or arraylike (numpy)
        Input values that are processed by the deadzone function
    para : dict
        Parameter dict with the following keys:
        slope_pos slope_neg threshold_pos threshold_neg

    Returns
    -------
    ndarray
        The signal after applying the deadzone operation, with values in
        the specified threshold range set to zero, and values above shifted by
        the threshold range.
    """
    mn = para['slope_neg']
    mp = para['slope_pos']
    tn = para['threshold_neg']
    tp = para['threshold_pos']

    d_out = (mp*(d_in - tp) * (d_in > tp) +
             0              * ((tn <= d_in) & (d_in <= tp)) +  # noqa
             mn*(d_in - tn) * (d_in < tn))
    return d_out
