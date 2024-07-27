<br />
<div align="center">
<h1 align="center">MuStar Research Project</h1>
  <h3 align="center">
    Defining the scalar mass relations between MuStar and the masses of galaxy clusters
from data samples selected by the South Pole Telescope (SPT)
  </h3>
</div>

<div align="center">
<img width="249" alt="image" src="https://github.com/user-attachments/assets/d965ee54-ad28-417f-8d4a-9b18efa32a2a">
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#background">Background</a></li>
    <li><a href="#scientific-motivation">Scientific Motivation</a></li>
    <li><a href="#languages-and-tools">Languages and Tools</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#datasets-and-parameter-limits">Datasets & Parameter Limits</a></li>
    <li><a href="#mathematical-formalism">Mathematical Formalism</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#mustar-predictions">MuStar Predictions & Discussion</a></li>
    <li><a href="#conclusions">Conclusions</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Background
<p align="left">
<b>What is a galaxy cluster?</b><br>
It is the largest gravitationally-bounded structure in the universe in equilibrium. They’re not only composed of an ensemble of galaxies. They’re mainly made of dark matter, ~82%, and a hot gas, 15%, while only 3% is stars.<br><br>
<b>What is MuStar?</b><br>
The sum of the galaxies’ stellar mass in a galaxy cluster or μ*<br><br>
<b>How can you weigh a dark matter halo with Galaxy Cluster optical images?</b><br>
By weighing the galaxy's stellar masses or via MuStar. More specifically, with a scaling relation we can measure the cluster mass with a given amount of uncertainty.<br><br>
<b>For what purpose?</b><br>
To measure the matter fraction of the universe and other cosmological parameters.<br><br>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Scientific Motivation

This was a collaborative project with Johnny Esteves at the University of Michigan within the Dark Energy group organized by Professor Marcelle Soares-Santos.

We, along with other scientists within The Dark Energy Survey Collaboration and The South Pole Telescope Collaboration, were focused on achieving three main goals:
- <b>Measure the stellar-to-halo mass relation</b> for SPT Clusters within the DES footprint
- To <b>measure and optimize</b> the precision of MuStar
- For cosmology, we want to <b>understand the effectiveness of using MuStar </b>by comparing it to other mass proxies (ex. Richness λ)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Languages and Tools
[![C++][cplusplus]][cplusplus-url]
[![Python][python]][python-url]
[![Jupyter Notebook][jupyter]][jupyter-url]
[![SciPy][scipy]][scipy-url]
[![Pandas][pandas]][pandas-url]
[![Numpy][numpy]][numpy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Methods
<p align="center">
We implemented a specific linear relation outlined in Farahi et al. 2019 as follows in order to model the relationship between the SZ signal ζ and MuStar μ<sup>*</sup>
</p>
<p align="center">
Simplified Linear Model<sup>3</sup>
</p>
<p align="center">
<img width="206" alt="image" src="https://github.com/user-attachments/assets/15feb096-bac7-4c83-bd52-e15ed6b39c72">
</p>
<p align="center">
Figure 3. Linear prediction for the values of the specified mass proxies (MuStar μ<sup>*</sup> and SZ signal ζ) and using the log mass of a cluster
</p>
<br>

<p align="center">
In this model assuming a first-order expansion of the halo-mass-function, the simplified version of the mass variance6 is:
</p>
<p align="center">
<img width="520" alt="image" src="https://github.com/user-attachments/assets/8c4a26ca-9833-41f0-b711-c823f8cdae6c">
</p>
<br>

<p align="center">
And the slope with the halo mass can be inferred by the SZ slope:
</p>
<p align="center">
<img width="168" alt="image" src="https://github.com/user-attachments/assets/a17785ff-a370-4916-8c28-697fe3f80620">
</p>
<br>

<p align="center">
The Richness λ was used as a proxy throughout our validation phase to confirm that the various implementations of our model were producing precise and accurate predictions (see validation section)
</p>
<p align="center">
As a result, we developed MCMC code to fit the relation by fitting a linear model of the Richness λ vs. SZ signal ζ, which will later be used to test MuStar μ<sup>*</sup>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Datasets & Parameter Limits

Datasets:
  - Dark Energy Survey Year 3 (Optical)
  - SPT 2500 deg2 SPT-SZ Survey

Parameter Limits:
  - Halo Mass range: 4.3 to 5.2 M☉ (solar masses)
  - Redshift range: 0.1 to 1.2

<p align="left">
SZ Signal<sup>4</sup>
</p>

Parameter | Constraint
--- | ---
A<sub>SZE</sub> | 5.24 ± 0.85
B<sub>SZE</sub> | 1.534 ± 0.100 
C<sub>SZE</sub> | 0.465 ± 0.407
σ<sub>SZE</sub> | 0.161 ± 0.080 
ρ | N/A 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MATHEMATICAL FORMALISM -->
## Mathematical Formalism
<p align="center">
SPT Scaling Relation<sup>2</sup>
</p>
<p align="center">
<img width="509" alt="image" src="https://github.com/user-attachments/assets/1ac63b03-cc13-4922-ada3-2b419bd2b21f">
</p>
<p align="center">
Figure 4. The value of the log of MuStar μ* and SZ signal ζ given our four specified priors
</p>
<br>

<p align="center">
Four free parameters (priors):<br>
A<sub>SZE/λ</sub> - <b>amplitude</b> of the SZ signal and richness,<br>
B<sub>SZE/λ</sub> - <b>mass slope</b> of the SZ signal and richness,<br>
C<sub>SZE/λ</sub> - <b>redshift evolution</b> of the SZ signal and richness<br>
σ<sub>SZE/λ</sub> - <b>intrinsic scatter</b> of the SZ signal and the richness<br>
ρ: correlation coefficient<br>
</p>
<br>

<p align="center">
Linear Model – Bivariate Gaussian Likelihood<sup>3,4</sup>
</p>
<p align="center">
<img width="507" alt="image" src="https://github.com/user-attachments/assets/774eca9e-28fc-4513-ac13-60479bae9ecc">
</p>
<p align="center">
Figure 5.  A Bayesian approach to define the SHMR as outlined in Bleem et al. 2019
</p>
<br>

<p align="center">
Covariance Matrix<sup>2</sup>
</p>
<p align="center">
<img width="590" alt="image" src="https://github.com/user-attachments/assets/525d3258-507d-49e1-a1af-d65348ed09a5">
</p>
<p align="center">
Figure 6. Describes the correlated intrinsic scatter between the two observables SZ signal ζ and MuStar μ*
</p>
<br>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

### Validation
<p align="center">
<img width="419" alt="image" src="https://github.com/user-attachments/assets/c6d5b3db-dfb3-497e-96ba-82f70081108e"><br>
Figures 7 & 8. Data and best linear model fit for a simulated/SPT-ECS data set<br>
</p>

<p align="center">
Our methodology can explain the difference with Bleem et al. 2019 on the slope. For instance, we used only a first-order expansion of the HMF and didn’t consider the redshift evolution.
</p>

### Scatter Optimization
<p align="center">
<img width="433" alt="image" src="https://github.com/user-attachments/assets/651ee6ef-5f93-4e7a-87da-e67fd81ab8ff">
</p>
<p align="center">
Figure 9. Scatter at fixed MuStar for different correlation values
</p>

<p align="center">
<img width="660" alt="image" src="https://github.com/user-attachments/assets/9af96e94-117e-4604-b60a-7283f15bc5eb">
</p>
<p align="center">
Figures 10 & 11. Optimizing MuStar - the slope and scatter are shown as a function of the cluster aperture
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## MuStar Predictions & Discussion

- During the validation, our slope was found to be **slighter steeper than Bleem et al. 2019**. This is because our power expansions only apply to the 1st order, and thus, our results do not represent a full computation in contrast to Grandis et al. 2021
- Our **preliminary, simplistic prediction of the scatter was a range of 0.41 ± 0.06** and informs the accuracy of our prediction
- Optimizing the definition of MuStar, we found that **R200 is the aperture where MuStar minimizes the scatter and maximizes the slope**
- We expected the results of the MuStar-to-halo mass relation will be within the range of 0.26 ± 0.125
- The difference can be explained by the **additional variance introduced by the redshift evolution of the relation**
- In addition, the Halo Mass Function or **HMF has an impact on our fit since we did not take it into account**


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Conclusions

- Our results will help to **aid future cluster cosmological analysis using the DES Y3 photometry** (as DES Y3 is one of the major datasets that we will be analyzing)
- The result of our calculated SPT-halo-mass relation can also be later used as a **cross-check for various upcoming cosmological results**
- Our results also highlight both the **inherent strengths and weaknesses of using a stellar-mass observable** instead of richness (λ) as well as better **highlight what the differences are between MuStar and the richness within the MuStar-halo-mass relation**
- In the future, the results of this work will be used to **measure the dark matter fraction of the Universe and other constraining cosmological parameters**


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

Allen Pinjic - apinjic@umich.edu<br>
Johnny Esteves - johnny.udi@gmail.com<br>
Marcelle Soares-Santos - marcelle@brandeis.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REFERENCES -->
 ## References

<!-- * [U-M Project Description](https://eecs281staff.github.io/p1-back-to-the-ship/#command-line-options) -->

1) [Bocquet, S. et al. Astrophys. J., vol. 878, 1, 2019, p. 55](https://arxiv.org/pdf/1812.01679)

2) [E, Bleem L. et al. Astrophys. J. Suppl., vol. 247, 1, 2020, p. 25](https://iopscience.iop.org/article/10.3847/1538-4365/ab6993/pdf)

3) [Evrard, August E., et al. Mon. Not. Roy. Astron. Soc., vol. 441, 4, 2014, pp. 3562–69](https://arxiv.org/pdf/1403.1456)

4) [Grandis, S. et al. Mon. Not. Roy. Astron. Soc., vol. 504, 1, 2021, pp. 1253–72](https://arxiv.org/pdf/2101.04984)

5) [Palmese, A. et al. Mon. Not. Roy. Astron. Soc. vol. 493, 4, 2020, pp. 4591–606](https://arxiv.org/pdf/1903.08813)

6) [Fahari, A. et al. Mon. Not. Roy. Astron. Soc. vol. 490, 3, 2019, pp. 3341–54](https://arxiv.org/pdf/1903.08042)

7) [“Scaling Relations | COSMOS.” Astronomy.swin.edu.au, Swinburne University of Technology](https://astronomy.swin.edu.au/cosmos/S/Scaling+Relations)

8) DES collaboration, E. Suchyta, P. Melchior (OSU, CCAPP)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!--[product-screenshot]: images/screenshot.png-->
[cplusplus]: https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cplusplus-url]: https://cplusplus.com/

[python]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue
[python-url]: https://www.python.org/

[scipy]: https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white
[scipy-url]: https://scipy.org/

[pandas]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/

[numpy]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/

[jupyter]: https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/
