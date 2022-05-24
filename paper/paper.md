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
affiliations:
 - name: Institute of Cosmology \& Gravitation, University of Portsmouth, Dennis Sciama Building, Burnaby Road, Portsmouth, PO1 3FX, United Kingdom
   index: 1
date: 24 May 20122
bibliography: paper.bib

---

# Summary

This package uses importance sampling to estimate the probability of rare first-passage time events, or FPT for short, which is the time taken to cross some threshold during a Langevin process. This is done by introducing a bias to over sample the events of interest, then recording the relative probability (weight) of this path occurring without the bias, so the probability original process can be recovered by applying the weight. The required data analysis to reconstruct the original probability density function for binned FPT values is included. This package runs simulations of thousands of Langevin processes, such that the probability density of the FPTs can estimated from the peak of the distribution all the way down into the far tail. Even probabilities as rare as $10^{-40}$ can simulated.

For efficiency, both Cython and multiprocessing is used.

# Statement of need

First-passage time problems, can appear for many random processes, including modeling financial ruin for insurance purposes or the mean time for nuclear collisions, to name just a few. Although the statistics of the FPTs can often be calculated analytically, the probability of rare events requires numerical simulations. This can be computationally very expensive, as millions of simulations are required just to produce a few of the events of interest. Therefore, importance sampling is used.


While PyFPT is designed to solve general FPT problems resulting from Langevin processes, it was developed in the context of stochastic inflation. Inflation is a period of accelerated expansion of spacetime near the beginning of the universe. Large, but rare, perturbations from this period can later form primordial black holes, which are of great theoretical interest. These perturbations can be modeled using FPT processes [@Vennin:2015hra]. Directly simulating these rare events often requires supercomputers, while importance sampling greatly improves the numerical efficiency.


# Acknowledgements

For invaluable contributions to both developing and physical understanding, I would like to thank for my supervisors David Wands, Vincent Vennin, Kazuya Koyama and Hooshyar Assadullahi. For making the code into a package, I would also like to thank Coleman Krawczyk and Ian Harry. This work was supported by the Science and Technology Facilities Council [grant numbers ST/S000550/1 and ST/W001225/1].

# References
