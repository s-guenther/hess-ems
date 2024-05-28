#!/usr/bin/env python3
"""
Fuzzy-logic-based EMS Module

This module implements a Fuzzy-logic-based Energy Management Strategy (EMS)
for hybrid energy storage systems.

Main method/EMS implementation:

    fuzzy(power_in, energy_peak, controller=build_controller())
        That performs the calcuations for the respective input with the
        respective controller and returns the powers for base and peak storage
    build_controller(mf_in=None, mf_out=None, ruleset=None):
        That generates a fuzzy controller object that can be used for
        calculation based on the definition of the input and output
        membership functions as well as a ruleset. Std parameters are used
        if these are not provided. See below variables for more information

Details on the parameterization of this EMS can be found in the variables

    `STD_MF_IN_DEF`
        Input membership functions definition
    `STD_MF_OUT_DEF`
        Output membership functions definition
    `STD_RULES`
        Ruleset Matrix definition
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# STD MEMBERSHIP FUNCTION DEFINITION FOR INPUT is a dict of dicts holding
# vectors/number lists. keys of the outer dict hold the names of the
# variables (which are `pin` and `epeak` in the fuzzy-logic-ems context).
# The keys of the inner dict hold the name of the respective membership
# functions and the values are either a 3-element array, holding the support
# points for a triangular MF, or 4-element array, holding the support points
# for a trapezoidal MF.
STD_MF_IN_DEF = {
    'pin': {
        'high in': [0.4, 0.8, 1, 1],
        'low in': [0, 0.4, 0.8],
        'no flow': [-0.4, 0, 0.4],
        'low out': [-0.8, -0.4, 0],
        'high out': [-1, -1, -0.8, -0.4],
    },
    'epeak': {
        'very low': [0, 0, 0.1, 0.3],
        'low': [0.1, 0.3, 0.5],
        'good': [0.3, 0.5, 0.7],
        'high': [0.5, 0.7, 0.9],
        'very high': [0.7, 0.9, 1, 1],
    }
}


# STD MEMBERSHIP FUNCTION DEFINITION FOR OUTPUT is structured in the same
# way as `STD_MF_IN_DEF`. There is only one output variable `pbase`.
STD_MF_OUT_DEF = {
    'pbase': {
        'high discharge': [-1, -1, -0.8, -0.4],
        'low discharge': [-0.8, -0.4, 0],
        'no flow': [-0.4, 0, 0.4],
        'low recharge': [0, 0.4, 0.8],
        'high recharge': [0.4, 0.8, 1, 1],
    }
}

# STD RULESET MATRIX that assigns input `pin` (rows) and `epeak` (columns)
# the the output `pbase` (matrix elements)
STD_RULES = np.array([
    ['no flow',         'low recharge',     'high recharge',  'high recharge',  'high recharge'],
    ['low discharge',   'no flow',          'low recharge',   'high recharge',  'high recharge'],
    ['high discharge',  'low discharge',    'no flow',        'low recharge',   'high recharge'],
    ['high discharge',  'high discharge',   'low discharge',  'no flow',        'low recharge'],
    ['high discharge',  'high discharge',   'high discharge', 'low discharge',  'no flow'],
])


# Function that generates the controller

def build_controller(mf_in=None, mf_out=None, ruleset=None):
    """Generates a fuzzy controller object that can perform calculations (of
    type `skfuzzy.ctrl.ControlSystemSimulation`).

    It uses structered dict of dicts to define the membership functions (MFs),
    inputs and outputs. See STD_MF_IN_DEF, STD_MF_OUT_DEF and STD_RULES for
    further information.

    Parameters
    ----------
    mf_in : dict of dicts holding 3- or 4-element number-arrays, optional
        Defines names of input variable, names of MFs and MF support points,
        Default: STD_MF_IN_DEF
    mf_out : dict of dicts holding 3- or 4-element number-arrays, optional
        Defines names of output variable, names of MFs and MF support points
        Default: STD_MF_OUT_DEF
    ruleset : numpy array of strings, optional
        nxm array, where n is the number of MFs of input one and m of input
        two. Default: STD_RULES
    """
    if mf_in is None:
        mf_in = STD_MF_IN_DEF
    if mf_out is None:
        mf_out = STD_MF_OUT_DEF
    if ruleset is None:
        ruleset = STD_RULES

    # ##
    # ## Add Antecedent/Consequent definition and in/out membership Functions
    in_vars = dict()
    out_vars = dict()
    zip_vars_mf_conante = zip([in_vars, out_vars],
                               [mf_in, mf_out],
                               [ctrl.Antecedent, ctrl. Consequent])
    for inout_vars, mf_inout, ConAnte in zip_vars_mf_conante:
        # The outer layer decides if the following loops iterate through the
        # input membership functions/antecedent, or the output memberschip
        # function, the consequent
        for var, mf_def in mf_inout.items():
            # Here, `var` holds the name of either input variables (see
            # `STD_MF_IN_DEF`) or output variables (see STD_MF_OUT_DEF)
            # MFDEF holds in a dict the linguistic variables of the membership
            # functions of the in/out variable as key and the MF support
            # points as value
            minmax = _get_minmax_from_mfvals(list(mf_def.values()))
            inout_vars[var] = ConAnte(np.linspace(*minmax, 101), var)
            for mf_name, mf_vals in mf_def.items():
                # mf_name is the linguistic variable (e.g. low medium high)
                # of the membership function and mf_vals is a list of
                # support points for the MF
                if len(mf_vals) == 3:
                    mf_fcn = fuzz.trimf
                elif len(mf_vals) == 4:
                    mf_fcn = fuzz.trapmf
                else:
                    raise ValueError('MF Definition must have 3 or 4 values')
                inout_vars[var][mf_name] = \
                    mf_fcn(inout_vars[var].universe, mf_vals)

    # ##
    # ## Add Rule Definition
    rules = []
    in_var1, in_var2 = list(in_vars.values())
    out_var = list(out_vars.values())[0]
    in_labels1 = list(in_var1.terms.keys())
    in_labels2 = list(in_var2.terms.keys())
    for (i, j), out_label in np.ndenumerate(ruleset):
        in_label1 = in_labels1[i]
        in_label2 = in_labels2[j]
        rule = ctrl.Rule(in_var1[in_label1] & in_var2[in_label2],
                         out_var[out_label],
                         label=f'{i},{j}')
        rules.append(rule)

    # ##
    # ## Create controller
    con_sys = ctrl.ControlSystem(rules)
    con_sys_sim = ctrl.ControlSystemSimulation(con_sys)
    return con_sys_sim


def _get_minmax_from_mfvals(mfval_list_of_lists):
    """Crawls a list of lists holding numbers and determines the minimum or
    maximum value, respectively"""
    minval = np.min([np.min(vec) for vec in mfval_list_of_lists])
    maxval = np.max([np.max(vec) for vec in mfval_list_of_lists])
    return [minval, maxval]


# Function that evaluates the control inputs with a specified controller

def fuzzy(power_in, energy_peak, controller=build_controller()):
    """
    Fuzzy-logic-based Energy Management Strategy (EMS)

    A Fuzzy-logic controller with two inputs (input power and peak energy
    storage) and one output (base power). The other output (peak power) is
    the difference from input to base. Each in and output has 5
    triangular/trapezodial membership functions (trapezodial at edges). The
    support points are defined via `para`.
    The ruleset is fixed, as well as the aggregation method (bounded sum),
    implication method (min), and defuzzification method (center of gravity).

    Default parameters are used if `para` is not provided or incomplete, as
    defined in `STD_PARA` and described in `STD_PARA_DESCRIPTIONS`.

    Parameters
    ----------
    power_in : scalar or arraylike (numpy)
        Input power to be managed by the EMS.
    energy_peak : scalar or arraylike (numpy)
        Peak energy input
    controller : fuzzy controller object
        generated with `build_controller()`

    Returns
    -------
    base
        Power dispatched to base storage.
    peak
        Power dispatched to peak storage.
    """
    controller.input['pin'] = power_in
    controller.input['epeak'] = energy_peak
    controller.compute()

    base = controller.output['pbase']
    peak = power_in - base

    return base, peak
