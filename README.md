# Online Scheduling for Energy Minimization in Wireless-Powered Mobile Edge Computing

This repository contains the Python research implementation associated with
the IEEE WCNC 2022 paper **"Online Scheduling for Energy Minimization in
Wireless Powered Mobile Edge Computing."** It simulates a multi-access-point,
multi-device wireless-powered mobile edge computing (WP-MEC) system and
evaluates online scheduling policies for wireless power transfer, local
computation, and computation offloading.

> **Research snapshot.** This is the original simulation-oriented codebase,
> preserved and documented for research reference. It is not currently a
> bit-exact reproduction package for every number or figure in the paper.

## Related paper

Xingqiu He, Yuhang Shen, Xiong Wang, Sheng Wang, Shizhong Xu, and Jing Ren,
"Online Scheduling for Energy Minimization in Wireless Powered Mobile Edge
Computing," *2022 IEEE Wireless Communications and Networking Conference
(WCNC)*, pp. 1146-1151, 2022.

- [IEEE DOI: 10.1109/WCNC51071.2022.9771592](https://doi.org/10.1109/WCNC51071.2022.9771592)
- [Open-access preprint: arXiv:2104.10893](https://arxiv.org/abs/2104.10893)
- [DBLP record](https://dblp.org/rec/conf/wcnc/HeSWWXR22)

```bibtex
@inproceedings{he2022online,
  author    = {Xingqiu He and Yuhang Shen and Xiong Wang and Sheng Wang and
               Shizhong Xu and Jing Ren},
  title     = {Online Scheduling for Energy Minimization in Wireless Powered
               Mobile Edge Computing},
  booktitle = {2022 IEEE Wireless Communications and Networking Conference
               (WCNC)},
  pages     = {1146--1151},
  year      = {2022},
  doi       = {10.1109/WCNC51071.2022.9771592}
}
```

## Implemented model

The simulator represents wireless devices (WDs) served by multiple access
points (APs). In each time slot, a scheduling policy coordinates:

1. wireless power transfer from an AP;
2. local computation at each WD;
3. computation offloading and WD-to-AP association;
4. transmit power, CPU frequency, and time allocation;
5. queue and battery-state updates.

The code includes three policy paths:

| ID | Policy | Implementation |
| --- | --- | --- |
| `0` | Proposed WP-MEC online policy | `Algorithm.executeWPMEC()` |
| `1` | Local-computation-only baseline (LCO) | `Algorithm.executeLCO()` |
| `2` | Full-offloading baseline (FO) | `Algorithm.executeFO()` |

The proposed path uses Lyapunov-style queue and battery terms, closed-form
updates where available, numerical constrained solves, and Hungarian
assignment for WD-to-AP association.

## Repository map

| Path | Purpose |
| --- | --- |
| `Scheduler.py` | Main experiment driver and parameter sweeps. |
| `Algorithm.py` | Proposed policy and baseline implementations. |
| `Environment.py` | Queue, battery, channel, and arrival-state transitions. |
| `Parameter.py` | System parameters and optional topology generation. |
| `GenerateMap.py` | Generates reusable AP/WD topology files. |
| `Auxiliary.py` | Numerical helper functions. |

## Environment

The documented compatibility environment uses Python 3.11 with NumPy 1.24
and SciPy 1.10:

```bash
conda env create -f environment.yml
conda activate wpmec-scheduling
```

This environment has been created successfully on Windows. Module imports and
a reduced one-slot LCO check pass. The proposed WP-MEC path, FO path, and full
default sweep have not been validated as regression tests in this environment.

## Running the simulation

The default script runs the `M` sweep for the three policies and writes
`change_m.mat` in the repository root:

```bash
python Scheduler.py
```

`Scheduler.py` has no `if __name__ == "__main__"` guard, so importing it also
starts this sweep. The default workload evaluates five values of `M`, three
policies, and 500 time slots per policy/value pair; expect a long-running
numerical experiment rather than a quick-start command.

The experiment selector near the bottom of `Scheduler.py` uses the following
IDs:

- `1`: sweep the Lyapunov trade-off parameter `V`;
- `2`: sweep the number of wireless devices `N`;
- `3`: sweep the number of access points `M`.

For example, `type = {1, 2, 3}` runs all three sweeps. Each output MATLAB file
contains energy and queue-length series for the proposed policy, LCO, and FO.

Topology files are not consumed by the checked-in default experiment. To
generate artifacts for inspection or manual integration:

```bash
mkdir data
python GenerateMap.py
```

## Reproducibility notes

- `Parameter.py` initializes NumPy with seed `47`, but the three policies run
  sequentially and do not currently replay an identical pre-generated random
  trace.
- `Environment.step()` regenerates the complete uplink/downlink channel
  matrices inside the per-device loop. A single time slot therefore resamples
  them `N` times, and different device updates can observe different channel
  snapshots. This should be corrected before interpreting policy comparisons.
- Several code constants use numerical scaling for queues, batteries, and bit
  units. Treat the checked-in configuration as an implementation snapshot,
  not as a complete paper-parameter manifest.
- The repository does not include the original plotting script or a golden
  set of paper-figure outputs.
- The current implementation retains historical NumPy array-to-scalar
  operations. They are accepted by the documented NumPy version, but should be
  made explicit before adopting newer numerical-library releases.

## Scope and limitations

This code is intended for simulation and research inspection. It is not a
production MEC controller, does not communicate with physical APs or devices,
and has not been safety-qualified for deployment. Before using numerical
results in a publication, record the commit, environment, random trace,
configuration, and generated output files.

## License status

This historical repository does not yet include a software license. The
provenance and reuse terms of inherited code must be confirmed before a new
license can be applied. Until then, public availability should not be
interpreted as permission to redistribute or relicense the source.
