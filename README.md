# HESS EMS

A project that implements various energy management strategies (EMS) for
hybrid energy storage systems (HESS), featuring a framework for simulation and
testing with several time series.

## Associated Work

The accompanying paper detailing hess-ems is forthcoming. This project's
simulation and test framework utilizes work from:

https://github.com/s-guenther/estss \
https://github.com/s-guenther/hybrid 

Be aware that there are no explicit dependencies to these projects. The used
data from the first `estss` package is copied into this one and the data
generated from the second `hybrid` package is manually generated and also stored
within this project. I.e., reproducing the published results is possible without
dependency on these packages. Further, the introduced ems can be used without
dependency on these packages and testing and visualizing other time series
without dependency on the `hybrid` package is also possible. 

Note that explicit dependencies on these projects are absent. Data from `estss`
is included in this project, and manually generated `hybrid` data is also stored
here, allowing for result reproduction without dependencies. The EMS introduced
can be operated independently, and testing and visualizing other time series
without `hybrid` is feasible.

## Overview

Implemented hess-ems, each in a separate file, include:

- __deadzone__ ems in `deadzone.py`
- __filter__ ems in `filter.py`
- __fuzzy__ ems in `fuzzy.py`
- __model predictive ems__ in `mpc.py`
- __neural network__ ems in `neural.py`

Additionally, the project contains:

- `simulate.py` for simulating specific storage settings with specific hess-ems
- `reference.py` for data export and import to/from hybrid
- `timeseries.py` for data retrieval from estss
- `visualize.py` for standardized simulation results visualization
- `mockup.py` providing mockups for unimplemented features
- `experiment.py` for creating published results and additional showcases

## Requirements

- Python `3.10` or newer


## Installation

Install manually by cloning the repository, entering it, and running `setup.py`:

```shell
    git clone https://github.com/s-guenther/hessems
    cd hessems
    python3 setup.py install
```

Alternatively, install via `pip`:

```shell
    pip install hess-ems
```


## Getting Started

_tdb_


## Contributing

Contributions are welcome! Please feel free to create issues or submit pull
requests for improvement suggestions or code contributions.


## License

Licensed under GPL-3.0-only. This program is free software and can be
redistributed and/or modified under the terms of the GNU General Public License
as published by the Free Software Foundation, version 3. It comes without any
warranty, including the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for details.

Commercial usage under GPLv3 is allowed without royalties or further
implications. However, discussions regarding potential cooperations are
appreciated.


## Author

HESS EMS - hybrid energy storage system energy management strategies
Copyright (C) 2024\
Sebastian Günther\
sebastian.guenther@ifes.uni-hannover.de

Leibniz Universität Hannover\
Institut für Elektrische Energiesysteme\
Fachgebiet für Elektrische Energiespeichersysteme

Leibniz University Hannover\
Institute of Electric Power Systems\
Electric Energy Storage Systems Section

https://www.ifes.uni-hannover.de/ees.html


