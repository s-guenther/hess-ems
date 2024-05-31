#!/usr/bin/env python3
"""
Model-predictive-control-based EMS Module (mpc-based EMS)

Main method/EMS implementation:
    mpc()
    MPCModel()

Details on the parameterization of this EMS can be found in the variables
    `STD_PARA` and
    `STD_PARA_DESCRIPTIONS`.
"""

from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
import pyomo.environ as pyo

from hessems.util import update_std
from hessems.deadzone import deadzone


# Standard Parameters needed for the Filter/Lowpass HESS-EMS
STD_PARA = {
    # Free parameters
    "w1": 1e5,
    "w2": 0.995,
    "w3": 0.005,
    "ref": 0.5,
    # Fixed parameters
    "pbase_max": 0.5,
    "pbase_min": -0.5,
    "ppeak_max": 0.5,
    "ppeak_min": -0.5,
    "ebase_max": 1,
    "epeak_max": 1,
    "tau_base": 1e-5,
    "tau_peak": 1e-3,
    "eta_base": 0.93,
    "eta_peak": 0.97,
}

# Description of the standard parameters
STD_PARA_DESCRIPTIONS = {
    # Free parameters
    "w1": 1e-2,
    "w2": 1,
    "w3": 1,
    "ref": 0.5,
    "pred_type": "full",
    # Fixed parameters
    "pbase_max": 0.5,
    "pbase_min": -0.5,
    "ppeak_max": 0.5,
    "ppeak_min": -0.5,
    "ebase_max": 1,
    "epeak_max": 1,
    "tau_base": 1e-5,
    "tau_peak": 1e-3,
    "eta_base": 0.93,
    "eta_peak": 0.97,
}


def mpc(power_in, dtime, energy_base, energy_peak, para=None):
    """ """
    # get parameters
    para = update_std(para, STD_PARA)
    model = MPCModel(power_in, dtime, energy_base, energy_peak, para)
    model.build()
    model.solve()
    # returns base, peak, pred_pb, pred_pp, pred_eb, pred_ep
    return model.results_as_tuple()


class MPCModel:
    def __init__(self, pin, dt, ebase, epeak, para):
        # Input and last state
        self.pin = pin
        self.dt = dt
        self.ebinit = ebase
        self.epinit = epeak
        # Fixed Parameters
        self.taub = para["tau_base"]
        self.taup = para["tau_peak"]
        self.etab = para["eta_base"]
        self.etap = para["eta_peak"]
        self.pb_max = para["pbase_max"]
        self.pb_min = para["pbase_min"]
        self.pp_max = para["ppeak_max"]
        self.pp_min = para["ppeak_min"]
        self.ebmax = para["ebase_max"]
        self.epmax = para["epeak_max"]
        self.w1 = para["w1"]
        self.w2 = para["w2"]
        self.w3 = para["w3"]
        self.ref = para["ref"]
        self.model = None
        self.results = None
        self.solverstatus = None

    def build(self):
        # Model Instantiation and Index
        model = pyo.ConcreteModel(name="MPCModel")
        ind = pyo.Set(initialize=range(len(self.pin)), ordered=True)
        model.ind = ind

        # Add non-pyomo-object vars to model
        model.ebinit = self.ebinit
        model.epinit = self.epinit
        model.pin = self.pin
        model.dt = self.dt
        model.taub = self.taub
        model.taup = self.taup
        model.etab = self.etab
        model.etap = self.etap
        model.w1 = self.w1
        model.w2 = self.w2
        model.w3 = self.w3
        model.ref = self.ref

        # power vars
        model.base = pyo.Var(ind, bounds=(self.pb_min, self.pb_max))
        model.peak = pyo.Var(ind, bounds=(self.pp_min, self.pp_max))
        model.baseplus = pyo.Var(ind, bounds=(0, self.pb_max))
        model.baseminus = pyo.Var(ind, bounds=(self.pb_min, 0))
        model.peakplus = pyo.Var(ind, bounds=(0, self.pp_max))
        model.peakminus = pyo.Var(ind, bounds=(self.pp_min, 0))
        model.hybrid = pyo.Var(
            ind, bounds=(self.pb_min + self.pp_min, self.pb_max + self.pp_max)
        )
        # energy vars
        model.baseenergy = pyo.Var(ind, bounds=(0, self.ebmax))
        model.peakenergy = pyo.Var(ind, bounds=(0, self.epmax))

        # Add constraint that locks base/peak plus and minus
        model.con_lockbase = pyo.Constraint(ind, rule=_lock_base_power)
        model.con_lockpeak = pyo.Constraint(ind, rule=_lock_peak_power)
        model.con_lockhybrid = pyo.Constraint(ind, rule=_lock_hybrid_power)

        # Add integration constraint/model equation
        model.con_integrate_base = pyo.Constraint(ind, rule=_integrate_base)
        model.con_integrate_peak = pyo.Constraint(ind, rule=_integrate_peak)

        # Add objective
        model.obj = pyo.Objective(expr=_objective_expression(model))

        # Add to MPCModel object
        self.model = model

    def solve(self):
        solver = pyo.SolverFactory("ipopt")
        status = solver.solve(self.model)
        isvalid = status["Solver"][0]["Status"] == "ok"
        self.solverstatus = status
        if not isvalid:
            warn("Optimization unsuccessful")
        self.results = self._extract_data()

    def _extract_data(self):
        if self.model is None:
            return
        results = dict()
        varnames = [
            "base",
            "baseplus",
            "baseminus",
            "baseenergy",
            "peak",
            "peakplus",
            "peakminus",
            "peakenergy",
            "hybrid",
        ]
        for var in varnames:
            results[var] = list(getattr(self.model, var).get_values().values())
        return results

    def _init_solution(self):
        """Estimates an initial solution for the solver. Note that this is
        unused as providing it does not seem to make a difference in
        computation time.
        You can integrate the data into the source code of `.build()` by
        calling this method and adding for each variable `pe.Var()` the option
        `initialize=` with the adequate data."""
        if self.results is None:
            return self._estimate_init_from_input()
        else:
            return self.results()

    def _estimate_init_from_last(self):
        """This should return self.results but shifted by one timestep. The
        last missing timestep must be somehow extrapolated. Not correctly
        implemented right now as providing an initial solution does not seem to
        make a diffference in computation time."""
        return self.results

    def _estimate_init_from_input(self):
        """Call deadzone function to estimate initial solution."""
        dzpara = {
            "slope_pos": 1,
            "slope_neg": 1,
            "out_max": self.pp_max,
            "out_min": self.pp_min,
            "threshold_pos": self.pb_max,
            "threshold_neg": self.pb_min,
            "gain": 1e-1,
            "window_up": self.ref,
            "window_low": self.ref,
            "base_max": self.pb_max,
            "peak_max": self.pp_max,
            "base_min": self.pb_min,
            "peak_min": self.pp_min,
        }
        base, peak, *_ = deadzone(self.pin, self.epinit, dzpara)
        baseplus = [b if b > 0 else 0 for b in base]
        baseminus = [b if b <= 0 else 0 for b in base]
        peakplus = [b if b > 0 else 0 for b in peak]
        peakminus = [p if p <= 0 else 0 for p in peak]
        hybrid = [b + p for b, p in zip(base, peak)]
        baseenergy = np.cumsum(base) + self.ebinit
        baseenergy[baseenergy > self.ebmax] = self.ebmax
        baseenergy[baseenergy < 0] = 0
        baseenergy = list(baseenergy)
        peakenergy = np.cumsum(peak) + self.epinit
        peakenergy[peakenergy > self.epmax] = self.epmax
        peakenergy[peakenergy < 0] = 0
        peakenergy = list(peakenergy)
        init = dict(
            base=base,
            peak=peak,
            baseminus=baseminus,
            peakminus=peakminus,
            baseplus=baseplus,
            peakplus=peakplus,
            hybrid=hybrid,
            baseenergy=baseenergy,
            peakenergy=peakenergy,
        )
        return init

    def results_as_tuple(self):
        base = self.results["base"][0]
        peak = self.results["peak"][0]
        pred_pb = self.results["base"]
        pred_pp = self.results["peak"]
        pred_eb = self.results["baseenergy"]
        pred_ep = self.results["peakenergy"]
        return base, peak, pred_pb, pred_pp, pred_eb, pred_ep

    def plot(self):
        if self.results is None:
            warn('No results available. Return without doing anything.')
            return
        fig, axs = plt.subplots(2, 1, sharex=True)
        x = np.arange(1, len(self.pin)+1)*self.dt
        axs[0].step(x, self.results['base'])
        axs[0].step(x, self.results['peak'])
        axs[0].step(x, self.pin)
        axs[0].step(x, self.results['hybrid'])
        axs[0].step(x, np.zeros_like(x), color='k', zorder=0)

        axs[1].step(x, self.results['baseenergy'])
        axs[1].step(x, self.results['peakenergy'])
        axs[1].step(x, np.zeros_like(x), color='k', zorder=0)

        axs[1].set_xlabel('time')
        axs[0].set_ylabel('power')
        axs[1].set_ylabel('energy')
        axs[0].legend(['base', 'peak', 'in', 'hybrid'])


# Pyomo model rules


def _lock_base_power(model, ii):
    return model.base[ii] == model.baseplus[ii] + model.baseminus[ii]


def _lock_peak_power(model, ii):
    return model.peak[ii] == model.peakplus[ii] + model.peakminus[ii]


def _lock_hybrid_power(model, ii):
    return model.hybrid[ii] == model.base[ii] + model.peak[ii]


def _integrate_base(model, ii):
    dt = model.dt
    taub = model.taub
    etab = model.etab
    if ii == 0:
        lastenergy = model.ebinit
    else:
        lastenergy = model.baseenergy[ii - 1]
    expression = (
        model.baseenergy[ii]
        == lastenergy * (1 - dt[ii] * taub)
        + (etab * model.baseplus[ii] + 1 / etab * model.baseminus[ii]) * dt[ii]
    )
    return expression


def _integrate_peak(model, ii):
    dt = model.dt
    taup = model.taup
    etap = model.etap
    if ii == 0:
        lastenergy = model.epinit
    else:
        lastenergy = model.peakenergy[ii - 1]
    expression = (
        model.peakenergy[ii]
        == lastenergy * (1 - dt[ii] * taup)
        + (etap * model.peakplus[ii] + 1 / etap * model.peakminus[ii]) * dt[ii]
    )
    return expression


def _objective_expression(model):
    w1, w2, w3 = model.w1, model.w2, model.w3
    expression = (
        w1 * sum((model.pin[ii] - model.hybrid[ii]) ** 2 for ii in model.ind)
        + w2 * sum((model.peakenergy[ii] - model.ref) ** 2 for ii in model.ind)
        + w3 * sum(model.base[ii] ** 2 for ii in model.ind)
    )
    return expression
