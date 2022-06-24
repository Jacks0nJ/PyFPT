<div id="top"></div>



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Jacks0nJ/PyFPT/blob/main/LICENSE)



<!-- ABOUT THE PROJECT -->
## What is PyFPT?

<p align="center">
  <img src="https://github.com/Jacks0nJ/PyFPT/blob/main/docs/images/PyFPT_logo.png?raw=true" width="500"/>
</p>

PyFPT is Python/Cython package to run first-passage time (FPT) simulations using importance sampling.

This package will let you numerically investigate the tail of the probability density for first passage times for a general 1D Langevin equation.

The tail of the probability density is investigated using the method of [importance sampling](https://arxiv.org/abs/nucl-th/9809075), where a bias increases the probability of large FPTs, resulting in a sample distribution, which are then weighted to reproduce the rare events of the target distribution. This allows very rare events (normally needing supercomputers) to be simulate efficiently with just your laptop!


Note, it was originally developed to find the local number of e-folds in slow-roll stochastic inflation. As such, analytical functionality is also included for this particular problem in the [analytics module](https://pyfpt.readthedocs.io/en/latest/analytics.html).

<p align="right">(<a href="#top">back to top</a>)</p>

## Documentation
You can find the [latest documentation](https://pyfpt.readthedocs.io/en/latest/index.html) on PyFPT's ReadTheDocs page.

## Requirements

### Operating System

As PyFPT uses [Cython](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) to optimise the stochastic simulations, a C-compilier is required for installation. Therefore, PyFPT does not currently run (future releases hope to address this issue) on Windows directly. Windows uses can either install PyFPT on a virtual machine or use a cloud-based service such as [SciServer](https://www.sciserver.org/). 


Mac and Linux user should be able to directly install PyFPT, as these operating systems have a C-compiler. Do feel free to raise an issue or contact us if you have any issues.

### Packages
The following packages are required to run PyFPT

* [Cython](https://cython.readthedocs.io/en/latest/src/quickstart/install.html)
* [NumPy](https://numpy.org/install/)
* [SciPy](https://scipy.org/install/)
* [matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [mpmath](https://mpmath.org/doc/current/setup.html)
* [pandas](https://pandas.pydata.org/docs/getting_started/install.html)

Many of which are included in common Python distributions like [Anaconda](https://www.anaconda.com/products/distribution). You can check which packages you already have installed with `pip list`.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### User Guide

The [documentation](https://pyfpt.readthedocs.io/en/latest/index.html) contains a [user guide](https://pyfpt.readthedocs.io/en/latest/getting_started.html), whose code you can run yourself as interactive [Jupyter notebook](https://jupyter.org/install) by [downloading](https://github.com/Jacks0nJ/PyFPT/tree/main/User%20guides) them.


### Installation

The package can be installed by using the following command
```sh
pip install PyFPT
```
in the command line wherever you have Python installed.

You can also clone the PyFPT repository

```sh
git clone https://github.com/Jacks0nJ/PyFPT.git
```
to work on it locally. This would require compiling the Cython code (the `.pyx` files) locally as well.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
### Example Results

The PyFPT package can be used to investigate far into the tail of the probability density (down to 10^-34 and beyond!)

<p align="center">
  <img src="https://github.com/Jacks0nJ/PyFPT/blob/main/docs/images/overlap_plot_m_0.001_bias_range_log_0.0_to_4.0_phi_UV_100.0phi_i_for_docs.png?raw=true" width="500"/>
</p>

Or even deviations from Gaussianity!

<p align="center">
  <img src="https://github.com/Jacks0nJ/PyFPT/blob/main/docs/images/publishable_error_bar_IS_near_10_dN_0.002_m_0.1_phi_UV_1.0phi_i_bias_3_iters_198234_bin_size_400_for_docs.png?raw=true" width="500"/>
</p>

In the above images `N' is the first-passage time in stochastic inflation.

See the [user guides](https://pyfpt.readthedocs.io/en/latest/getting_started.html) for details on how you can make these figures yourself!


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Simulate first-passage times of slow-roll inflation
- [x] Use importance sampling to investigate rare realisations.
- [x] Make general, for any 1D Langevin equation
- [ ] Add multi-dimensionality
    - [ ] Add the acceleration of the field
    - [ ] Add more sophisticated noise models

See the [open issues](https://github.com/Jacks0nJ/PyFPT/issues) for a full list of known issues.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

And we will review your request!

<p align="right">(<a href="#top">back to top</a>)</p>

## Bugs
This is the initial release of PyFPT, so it is expected there will be some minor bugs. Feel free to either report by raising an [Issue](https://github.com/Jacks0nJ/PyFPT/issues) on Github, emailing joseph.jackson@port.ac.uk or fork the repository with your fix. 

Your feedback is very much appreciated! 



<!-- LICENSE -->
## License

Distributed under an Apache-2.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Joe Jackson  - joseph.jackson@port.ac.uk

Project Link: [https://github.com/Jacks0nJ/PyFPT](https://github.com/Jacks0nJ/PyFPT)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like the following contributors to PyFPT, be it through physical understanding of first-passage time processes or help developing the package

#### The Physics

* David Wands
* Vincent Vennin
* Kazuya Koyama
* Hooshyar Assadullahi

#### Package Development

* Coleman Krawczyk
* Ian Harry

#### Logo

* Will Jackson

#### Resources

The following resoucres were instrumental in developing the project into a package usable by the community:

* [Sphinx Guide](https://techwritingmatters.com/documenting-with-sphinx-tutorial-part-1-setting-up)
* [Autodoc](https://eikonomega.medium.com/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365)
* [Package Development](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
* [README.md](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png

