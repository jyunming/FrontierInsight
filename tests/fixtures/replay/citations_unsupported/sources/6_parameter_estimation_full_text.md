---
auto_collected: true
rank: 19
source: web_search
backend: brave
kind: web_page
title: Parameter Estimation of Complex Chemical Kinetics with ...
url: https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_469-476.pdf
snippet: "The parameter estimation is started and CMA-ES sends the proposed coefficients\
  \ for \xB7 parameters to in-house driver program. The SENKIN module simulates the\
  \ model with \xB7 parameter values, which are calculated to be the product of the\
  \ proposed coefficient and \xB7 literature value of the respective kinetic parameter."
site: match.pmf.kg.ac.rs
fetched_full_text: true
full_text_bytes: 16647
screen_grade: 1
content_quality: full_text
retraction: no_doi
source_text_scanned: 2:17784::0:0
---
Parameter Estimation of Complex Chemical Kinetics with ...

The parameter estimation is started and CMA-ES sends the proposed coefficients for · parameters to in-house driver program. The SENKIN module simulates the model with · parameter values, which are calculated to be the product of the proposed coefficient and · literature value of the respective kinetic parameter.

---FULL TEXT (fetched)---

Parameter Estimation of Complex Chemical
Kinetics with Covariance Matrix Adaptation
Evolution Strategy

Petteri Suominen1*, Anders Brink 2, and Tapio Salmi1

1Laboratory of Industrial Chemistry and Reaction Engineering, Process Chemistry Centre,
Åbo Akademi University, Biskopsgatan 8 FI-20500 Turku, Finland
2Laboratory of Inorganic Chemistry, Process Chemistry Centre, Åbo Akademi University

e-mails: timo.suominen@abo.fi, {tapio.salmi, anders.brink}@abo.fi

(Received February 10, 2012)
Abstract
This paper presents a method for parameter estimation of complex chemical kinetics
by an evolution strategy which uses a scheme called covariance matrix adaptation. The
advantage of this scheme is that a completely derandomized self-adaptation of mutation
distribution can be achieved. The used algorithm utilizes even cumulation to improve
the performance.
The method was tested on experimental data of ethene pyrolysis and the parameters
of two kinetic models depicting the phenomenon were estimated. The models
comprised of 37 and 687 reversible reactions respectively. The method successfully
estimated the kinetic parameters for both models.

1 Introduction
Evolution algorithms are effective and robust methods for optimization. Their biggest
disadvantage is the poor convergence performance and therefore they are most efficient in
global optimization problems that have many local optima. They have been used in parameter
estimation of complex chemical kinetics, but have not gained popularity over conventional
gradient/derivative based methods, which are computationally lighter in simpler parameter
estimation tasks. In large parameter estimation tasks, however, evolution algorithms have a
clear advantage as the choice of initial parameter values is by far simpler task as it is for
gradient/derivative based methods.
MATCH
Communications in Mathematical
and in Computer Chemistry
MATCH Commun. Math. Comput. Chem. 68 (2012) 469-476

ISSN 0340 - 6253

Evolution algorithms can be divided in different techniques based on the implementation
details. These techniques include genetic algorithm, genetic programming, evolution
programming, neuroevolution and evolution strategy. For this paper, an evolution strategy
(ES) was chosen for parameter estimation. The selection process in evolution strategies is
based on the fitness rankings rather than the actual fitness values and it is deterministic. This
is one of the main reasons for the robustness of ESs. Besides selection, mutation is used as a
search operator. The step-size of mutation is often governed by self-adaptation to keep the
progress in the evolution window. Another technique to ensure the convergence to the
optimum of a function is cumulative step size adaptation.
Evolution strategies have been previously used in parameter estimation of complex
chemical kinetics by Polifke et al [1]. In that work, a simple (μ + λ)-ES by Rechenberg et al.
[2] was used. Some improvements to this work has been proposed by Elliott el al [3]
including addition of recombination operation, which is occasionally used in evolution
strategies to prevent the algorithm to stuck on local optima. Other types of evolution
algorithms, mainly genetic algorithms, and also other types of stochastic algorithms, such as
particle swarm and differential evolution (DE) have also been applied to parameter estimation
in chemical kinetics.
To our work, an evolution strategy with covariance matrix adaptation (CMA-ES) by
Hansen and Ostermeier [4] was chosen. Hansen and Ostermeier depict the step to CMA-ES
from an ES with isotropic mutation distribution as comparable to a step from a simple
gradient-based method to a quasi-Newton method, e.g. a step from local gradient to
approximation of inverse Hessian matrix. CMA-ES has already been applied to many real-
world search problems. The advantage of CMA-ES over the conventional ESs is the added
invariance properties.
The detailed models of tar formation from light hydrocarbons are very complex,
consisting of several hundreds of reactions. As the amount of estimated parameters can be as
high as three times the amount of reactions, parameter estimation of chemical kinetics for tar
formation is a practical example of an optimization problem, for which conventional
gradient/derivative based algorithms are difficult to implement. It has been proposed that the
limit above which stochastic optimization methods outperform the gradient based methods is
as low as ten dimensions.
-470-

2
Parameter estimation problem and software
2.1 Parameter estimation problem
The chemical system which the algorithm was tested on is the early stage of tar formation,
where heavier gaseous hydrocarbons are formed from light hydrocarbons at elevated
temperatures. Two models, which depict the phenomenon, were subject to parameter
estimation. One was a detailed kinetic model from Lawrence Livermore National
Laboratories (LLNL) [5] and the other one was a reduced kinetic model of our own [6]. The
phenomenon is complex and even the reduced model comprises of 37 reversible reactions and
30 species. Both models do give as such a decent fit to experimental data, but to obtain a good
fit to experimental data, parameter estimation offered an effective method.
The utilization of conventional parameter estimation algorithms, which are based on
derivatives and gradients, would be tedious to implement to a differential equation system of
this magnitude because the topology of the surface is detailed. An additional challenge is that
parameter estimation for this kind of system is a non-trivial to solve as the rate constants are
typically of different magnitude, some of the numerical values are close to the machine
precision limit, while others are very large and due to the intrinsic feature of chain-reaction
kinetics, for which the rates change rapidly due to branching reactions. These dilemma need
to be solved.

2.2 Evolution strategy with covariance matrix adaptation (CMA-ES)
CMA-ES is an evolution strategy which implements two important self-adaptation methods,
namely derandomization and cumulation. With cumulation, the information from the past
generations is used in the self-adaptation by considering the search-path the population has
undergone. Derandomization is a process, in which the mutation distribution is changed
deterministically so that the probability of the previously successful steps is increased.
The CMA-ES has been tested on multi-modal test functions by Hansen and Kern [7]. In
their work, the performance of the CMA-ES algorithm is compared to the performance of the
differential evolution algorithm, the Robust Evolution Strategy (RES) and Local Optima
Smoothing (LOS) restart BFGS algorithm. Only on additively separable functions, the CMA-
ES was outperformed by the DE algorithm. Therefore, CMA-ES is seemingly a plausible
candidate for an effective parameter estimation algorithm for complex chemical kinetics.
-471-

According to Hansen and Ostermeier [4], roughly 10n function evaluations are needed for
a significant change in mutation distribution shape and strategy behaviour cannot realistically
be expected to improve until 30n function evaluations have been run. A complete adaptation
can take even 100n2 function evaluations and between 100 and 200(n+3)2 function
evaluations are recommended to get maximal advantage from the self-adaptation.

2.3 Parameter estimation software
As the parameter estimation software, which we have been previously been using, failed to
perform parameter estimation on the system described in section 2.1, we were forced to create
in-house parameter estimation software. This software was required to be able to perform the
parameter estimation on complex chemical kinetics. There are program codes, which can
simulate gas-phase radical chemistry. It is obvious that if such software could be modified to
do the parameter estimation, a lot of time could be saved. CHEMKIN II [8] has been designed
primarily to simulate gas-phase radical chemistry and the numerical issues intrinsic to this
kind of system have been successfully solved already. As CHEMKIN II was distributed with
the source code, it offers a flexible alternative for implementing some in-house code. In our
case, a post-processor and a driver program were needed. Therefore, it was chosen to be used
in our parameter estimation software.
Other choice to be made was the optimization algorithm. According to literature findings,
stochastic algorithms have been effective in parameter estimation of such problems. Due to
the reasons presented in section 2.2, a CMA-ES algorithm by Hansen and Ostermeier [4] was
selected.
The used software consists of SENKIN [9] module of CHEMKIN II software package,
CMA-ES algorithm from Hansen and Ostermeier and a post-processor of our own. The
software is schematically illustrated in Figure 1.

The initial strategy parameters are manually inserted to CMA-ES algorithm. Likewise,
the model is written to an input file to the SENKIN module of the CHEMKIN II software
package. The parameter estimation is started and CMA-ES sends the proposed coefficients for
parameters to in-house driver program. The SENKIN module simulates the model with
parameter values, which are calculated to be the product of the proposed coefficient and
literature value of the respective kinetic parameter. After a successful simulation, the in-house
driver starts the in-house post-processor, which calculates the sum of squares of the simulated
-472-

values and our experimental results. The post-processor returns this value to driver which
returns it to CMA-ES algorithm as a fitness value. At the same time, both in-house modules
record the all-time best values. CMA-ES then proposes new values for coefficients, which
undergo the same process until a termination condition is met. When a termination condition
is met, CMA-ES produces the results of its workings. The in-house driver produces a
CHEMKIN input file with best found parameter values and post-processor gives results in a
form, which can easily be plotted graphically.



Figure 1: A schematic view of the parameter estimation software

3 Results and discussion
The goodness of the fit was estimated by the sum of squares, which was calculated from
experimental results of our own. Kinetic experiments were conducted at six temperatures
from 500 °C to 1000 °C and for two different compositions. Both mixtures contained 2 mol-%
ethene and 10 mol-% methane and the other one had 35 mol-% of hydrogen. Nitrogen was
used as a balance gas. All in all, 120 experimental points were recorded for the calculation of
sum of squares. The experimental setup is presented in detail in another article [10]. In case of
-473-

a detailed kinetic model from LLNL, the sum of squares was lower, even though the
magnitude was the same as with our reduced kinetic model prior to the parameter estimation.
A regression analysis of the model predictions and the experimental values was
performed by the data analysis toolpack in Excel. It was observed that the R2-value, the
coefficient of determination, increased significantly by the parameter estimation and was
0.966 after the parameter estimation. The R2-value is a measure of how likely the model
predicts the future outcomes. The sum of square values in both cases were over 10 % lower
after the parameter estimation compared to prior the parameter estimation. Graphical
illustrations of the experimental values compared to model predictions prior to and after
parameter estimation are presented in Figure 2a-b.



(b) Model predictions after parameter estimation
Figure 2: Graphical illustration of the model predictions (a) prior to parameter

In case of the LLNL model, the advantage of the adaptation of the mutation distribution
could not be observed as the parameter values were not allowed to change more than 5 % and
the best combination of parameter values was found at less than 30n function evaluations
which is according to Hansen and Ostermeier the limit for the adaptation to improve the
strategy behaviour. In the case of our reduced model, more evaluations were required to
obtain a solution with an acceptable accuracy. Therefore, the adaptation did have time to
evolve. The same restriction of 5 % was applied even here. Even in this case, the full
advantage of the adaptation as the limit of 100n2 function evaluations was not reached.
During the development of our model, we tried to perform the parameter estimation on
preliminary models, too. In a case when the estimated parameters were not close to the
literature findings, considerably more function evaluations were needed to reach a satisfactory
-474-

accuracy for the model. In attempts like these, no restrictions were applied. This indicates that
CMA-ES could be an interesting candidate for parameter estimation in such applications,
where the magnitudes of the parameters are unknown and the amount of parameters is large.
Possible applications are problems, in which diffusion and/or adsorption effects are added to
complex chemical kinetics.
Hansen and Kern [7] have reported that increasing the population, in many cases,
considerably increases the performance of the algorithm. In our applications, this kind of
behaviour could not be observed in significant magnitude. As some test functions in their
report did not either exhibit such behaviour, this cannot considered a surprising result. It is
though possible that the population size should have been increased even more than we did to
obtain an increase in algorithm performance.
Different parameter estimation approaches were tested. We tried to estimate all the
parameters in the modified Arrhenius equation presented in Eq. 1 and different combinations
of them. The most efficient way was to estimate only one parameter. Intuitively, this is clear
as the size of covariance matrix increases but the amount of data from which it is estimated
does not increase.

*
0
*
*
a
n
E
R T
T
k
A
e
T









(1)

4 Conclusions
The software successfully estimated the parameters of the complex chemical kinetics. The
modeled phenomenon was depicted by two models, a semi-large model of 37 reversible
reactions and a large model comprising 687 reversible reactions. In both cases the sum of
squares was significantly lower after the parameter estimation than it was prior to the
parameter estimation.
The adaptation of covariance matrix could not shown to have an importance in case of a
detailed kinetic model, in which the parameter values were allowed to change only a little,
while the values of parameters were close to literature values. On the other hand, when the
magnitude of the parameter values was unknown, CMA-ES did offer an efficient alternative
for the parameter estimation.

-475-

References
[1]
W. Polifke, W. Geng, K. Döbbeling, Optimization of rate coefficients for simplified
reaction mechanisms with genetic algorithms, Combust. Flame 113 (1998) 119–135.
[2]
I. Rechenberg, Evolutionsstrategie ’94, Frommann–Holzboog, Stuttgart, 1994.
[3]
L. Elliott, D. Ingham, A. Kyne, N. Mera, M. Pourkashanian, C. Wilson, Genetic
algorithms for optimisation of chemical kinetics reaction mechanisms, Prog. Energ.
Combust. Sci. 30 (2004) 297–328.
[4]
N. Hansen, A. Ostermeier, Completely derandomized self–adaptation in evolution
strategies, Evol. Comput. 9 (2001) 159–195.
[5]
N. Marinov, W. Pitz, C. Westbrook, A. Vincitore, M. Castaldi, S. Senkan, Aromatic
and polycyclic aromatic hydrocarbon formation in a laminar premixed n-butane flame,
Combust. Flame 114 (1998) 192–213.
[6]
P. Suominen, T. Salmi, A. Brink, A reduced reaction mechanism for light hydrocarbon
thermal reactions – estimation of kinetic parameters, submitted.
[7]
N. Hansen, S. Kern, Evaluating the CMA evolution strategy on multimodal test
functions, in: Xin Yao, Edmund K. Burke, José A. Lozano, Jim Smith, Juan Julián
Merelo-Guervós, John A. Bullinaria, Jonathan E. Rowe, Peter Tino, Ata Kabán, Hans-
Paul Schwefel (Eds.), Parallel Problem Solving from Nature - PPSN VIII, Springer–
Verlag, Berlin, 2004, pp. 282–291.
[8]
R. Kee, F. Rupley, J. Miller, Chemkin-II: A Fortran Chemical Kinetics Package for
the Analysis of Gas Phase Chemical Kinetics, Sandia National Labs., Livermore,
1989.
[9]
A. E. Lutz, R. J. Kee, J. A. Miller, SENKIN, A Fortran Program for Predicting
Homogeneous Gas Phase Kinetics with Sensitivity Analysis, Sandia National Labs.,
Livermore, 1988.
[10]
P. Suominen, M. Kangas, T. Salmi, P. Simell, M. Reinikainen, Thermal reactions of
the main hydrocarbon components in gasification gas, submitted.


-476-

---VALUES READ FROM THE FIGURES (by a model, from the images)---

Figure 2 (page 6): Figure 2: Graphical illustration of the model predictions (a) prior to parameter estimation (b)
Read from the image: Two side-by-side plots (a and b) at 1000 °C. x-axis: Residence time (s) from 0 to 8; y-axis: mass fraction from 0 to 0.1. Series: CH4 (yellow) is highest, starting at about 0.09 and remaining nearly constant; H2 (red) is constant at about 0.04; C2H6 (blue) starts at about 0.03 and decays to about 0.015 by 2s; C2H4 (cyan) starts at 0 and rises to about 0.015 by 2s; C2H2 (green) starts at 0 and rises to about 0.01 by 2s; C2H4 (dark blue) remains near 0. The plots show model predictions before and after parameter estimation, with data points plotted as dots.

## Figures

- **Figure 1**, page 5: `figures/lit_019_fig1_p5.png`  
  Figure 1: A schematic view of the parameter estimation software
- **Figure 2**, page 6: `figures/lit_019_fig2_p6.png`  
  Figure 2: Graphical illustration of the model predictions (a) prior to parameter estimation (b)
