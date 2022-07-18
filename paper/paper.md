---
title: 'PyFPT: A Python package for first-passage times'
tags:
  - Python
  - Cython
  - first-passage times
  - importance sampling
  - cosmology
  - inflation
authors:
  - name: Joseph Jackson
    affiliation: 1
  - name: Ian Harry
    affiliation: 1
  - name: Coleman Krawczyk
    affiliation: 1
affiliations:
 - name: Institute of Cosmology \& Gravitation, University of Portsmouth, Dennis Sciama Building, Burnaby Road, Portsmouth, PO1 3FX, United Kingdom
   index: 1
date: 24 May 20122
bibliography: paper.bib

---

# Summary

This package uses importance sampling to estimate the probability of rare first-passage time events, or FPT for short. First-passage time problems can appear for many random processes, including pricing options for finance or the mean time for nuclear collisions, to name just a few. The FPT is the time taken to cross a threshold during a ~~Langevin~~ random process [@Redner2001qk]. Although the statistics of FPTs can often be calculated analytically, the probability of very rare events requires numerical simulations. This can be computationally very expensive, as millions of simulations are required just to produce a few of the events of interest. Therefore, importance sampling is used. This is done by introducing a bias to oversample the events of interest, then recording the relative probability (weight) of this path occurring without the bias, so the probability of the original process is recovered [@Mazonka1998]. The required data analysis to reconstruct the original probability density function for binned FPT values is included. This package runs simulations of thousands of Langevin processes, such that the probability density of the FPTs can be estimated from the peak of the distribution all the way down into the far tail. Even probabilities as rare as $10^{-40}$ can be simulated.

For performance optimization, both [Cython](https://cython.org/) and [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) are used.

# Statement of need

While PyFPT is designed to solve general one-dimensional FPT problems resulting from Langevin processes, it was developed in the context of stochastic inflation. Inflation is a period of accelerated expansion of spacetime near the beginning of the universe [@Baumann:2009ds]. Large, but rare, perturbations from this period can later form primordial black holes, which are of great theoretical interest [@Green:2020jor]. These perturbations can be modelled using FPT processes [@Vennin:2015hra]. Directly simulating these rare events often requires supercomputers, while with importance sampling only a single CPU is required.

# State of Field

To the authors' knowledge, this is the first open-source program simulating FPT problems using importance sampling. While there are many open-source codes to [solve stochastic differential equations](https://github.com/topics/stochastic-differential-equations), they are not specialised to realising rare FPT events in Langevin processes. Conversely, there are also codes which do solve FPT problems in stochastic inflation, but they are not open-source, see for example [@Figueroa:2021zah] and [@Mahbub:2022osb].


# Acknowledgements

For invaluable contributions to both developing and physical understanding, JJ would like to thank his supervisors David Wands, Vincent Vennin, Kazuya Koyama and Hooshyar Assadullahi. This work was supported by the Science and Technology Facilities Council [grant numbers ST/S000550/1 and ST/W001225/1].

# References
