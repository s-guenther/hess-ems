#!/usr/bin/env python3
"""
Lowpass/Filter-based EMS Module

This module defines a Lowpass/Filter Energy Management Strategy (EMS)
utilizing a first-order lowpass filter combined with proportional energy
feedback for hybrid energy storage systems.

Main method/EMS implementation:

lowpass(power_in, dtime, last_filt, energy_peak, para=None)
    Applies a first-order lowpass filter to input power, incorporating
    proportional energy feedback based on peak energy levels. Supports
    customization through parameters, with defaults provided if unspecified.

Details on the parameterization of this EMS can be found in the variables
    `STD_PARA` and
    `STD_PARA_DESCRIPTIONS`.
"""

import numpy as np

from hessems.util import update_std


# Standard Parameters needed for the Filter/Lowpass HESS-EMS
STD_PARA = {
    'fcut': 1e-2,
    'gain': 1,
    'ref': 0.5
}

# Description of the standard parameters
STD_PARA_DESCRIPTIONS = {
    'fcut': 'Cut-off frequency of the 1st order low pass filter',
    'gain': 'feedback gain of the soc feedback controller',
    'ref': 'Reference energy of to compare to in the feedback loop'
}


def lowpass(power_in, dtime, last_filt, energy_peak, para=None):
    """
    Filter/Lowpass Energy Management Strategy (EMS)

    This EMS applies a discrete first-order lowpass filter to the input power, 
    with proportional feedback based on peak energy. The filter 
    smoothens fluctuations in power input and dispatches the smoothed result
    to the base storage (minus the energy feedback from peak storage).
    
    Default parameters are used if `para` is not provided or incomplete, as
    defined in `STD_PARA` and described in `STD_PARA_DESCRIPTIONS`.

    Parameters
    ----------
    power_in : scalar/float
        The input power to be dispatched to base and peak storage
    dtime : scalar/float
        The time difference from the current to the last step.
    last_filt : scalar/float
        The last internal filter value, necessary for integration.
    energy_peak : scalar/float
        Peak energy input for feedback calculation.
    para : dict, optional
        A dictionary of parameters for the lowpass filter. Expected keys:
        'fcut' (filter cutoff frequency), 'gain' (feedback gain), 
        and 'ref' (reference value for feedback).

    Returns
    -------
    base : float
        Power dispatched to base storage.
    peak : float
        Power dispatched to peak storage.
    filt : float
        Current internal integrator value of the lowpass filter.
    feedback : float
        Current soc feedback value
        
    Note
    ----
    See 
    https://en.wikipedia.org/wiki/Low-pass_filter#Discrete-time_realization
    for more info on discrete lowpass implemenation
    """

    # get parameters
    para = update_std(para, STD_PARA)
    fcut = para['fcut']
    gain = para['gain']
    ref = para['ref']
    # rename input vars for easier equation writing
    pin = power_in
    dt = dtime
    ep = energy_peak
    last = last_filt
    pi = np.pi

    # convert filter cutoff fcut to alpha
    alpha = 2 * pi * dt * fcut / (2 * pi * dt * fcut + 1)
    # compute filter step
    filt = alpha * pin + (1 - alpha) * last
    # compute feedback
    feedback = (ep - ref) * gain

    # write output
    base = filt + feedback
    peak = pin - base
    return base, peak, filt, feedback
