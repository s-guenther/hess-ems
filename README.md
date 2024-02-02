# HESS-EMS

A project that implements various representative energy management strategies 
(EMS) for hybrid energy storage systems (HESS)

---

## Associated Work

The accompanying paper detailing HESS-EMS is forthcoming.

To streamline this project and minimize dependencies, a dedicated simulation
and testing framework for these EMS strategies has been established in a 
separate project, available at
[HESS-EMS-SIM](https://github.com/s-guenther/hessemssim).
This repository also hosts the code for generating the plots featured in the
forthcoming paper.


## Overview

Implemented hess-ems, each in a separate file, include:

- __deadzone__ ems in `deadzone.py`
- __lowpass__ ems in `lowpass.py` <sup>(*1)</sup>
- __fuzzy__ ems in `fuzzy.py`
- __model predictive__ ems in `mpc.py`
- __neural network__ ems in `neural.py`

<sub>
(*1) The original EMS name in the publication is `filter`, which is also 
more in line with the rest of the literature. However, this name would 
shadow the respective
[https://docs.python.org/3/library/functions.html#filter](python builtin function)
and we therefore chose to rename it to `lowpass` in this project.
</sub>
<br /><br />

The `examples/` folder contains simple visualizations and test cases of the 
implemented EMS. (These test cases can also be interpreted as simple,
nonautomated function tests)


See the source code documentation for further information.


## Requirements

Developed with Python `3.11`. Should work with way older versions since
`3.6` (f-strings) as well.


## Installation

Install manually by cloning the repository, entering it, and build and 
install via the `pyproject.toml`:

```shell
    git clone https://github.com/s-guenther/hessems
    cd hessems
    python3 -m build
    pip install .
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

HESS EMS - hybrid energy storage system energy management strategies\
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


