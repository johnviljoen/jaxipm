# E4 (existing data) — paired final-cost comparison, jaxipm vs IPOPT
Pairs are formed on the SAME problem instance (nav: start angle; track: start state x0 → pool index). IPOPT values come from the E1 P=1 sweep (tol 1e-6); jaxipm values from the paper-driver npz (tol 1e-8, ir_nsteps 0). Constraint-violation / KKT-residual columns need the full z (not saved by the paper drivers) and come from the ablation runs.


### nav 90°  [jaxipm solved 1633/2000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 1633)

- mean objective: jaxipm 517.861, IPOPT 517.942; median jaxipm 511.195, IPOPT 511.746
- mean paired difference -0.0804 (std 7.6323); median -0.67848; mean relative gap -0.013% (median -0.1332%)
- 95% bootstrap CI of the mean relative gap: [-0.085%, +0.061%]
- jaxipm lower by >0.0001 rel: 96.1%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 3.9%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -33.403%, -0.720%, -0.291%, -0.133%, -0.095%, +5.637%, +15.663%
- instances with |rel gap| > 1% (different local optimum or failed solve): 61 (3.74%)
- Wilcoxon signed-rank p = 3.61e-195; paired t-test p = 0.671 (t = -0.43)
- excluding |rel gap| > 1%: mean rel gap -0.1560%, Wilcoxon p = 3.98e-243

### nav 180°  [jaxipm solved 1634/2000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 1634)

- mean objective: jaxipm 507.009, IPOPT 507.803; median jaxipm 497.796, IPOPT 498.305
- mean paired difference -0.7941 (std 6.7160); median -0.66834; mean relative gap -0.157% (median -0.1326%)
- 95% bootstrap CI of the mean relative gap: [-0.230%, -0.107%]
- jaxipm lower by >0.0001 rel: 99.4%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.6%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.052%, -0.306%, -0.266%, -0.133%, -0.099%, -0.093%, +13.895%
- instances with |rel gap| > 1% (different local optimum or failed solve): 10 (0.61%)
- Wilcoxon signed-rank p = 5.36e-257; paired t-test p = 1.91e-06 (t = -4.78)
- excluding |rel gap| > 1%: mean rel gap -0.1521%, Wilcoxon p = 1.24e-265

### track v̄=1.0  [jaxipm solved 628/1000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 628)

- mean objective: jaxipm 254.284, IPOPT 268.011; median jaxipm 241.287, IPOPT 247.100
- mean paired difference -13.7272 (std 81.4833); median -0.00000; mean relative gap -2.161% (median -0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.803%, -0.135%]
- jaxipm lower by >0.0001 rel: 12.3%; equal within 0.0001: 78.3%; jaxipm higher by >0.0001: 9.4%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -73.573%, -62.925%, -32.468%, -0.000%, +1.359%, +20.071%, +359.623%
- instances with |rel gap| > 1% (different local optimum or failed solve): 111 (17.68%)
- Wilcoxon signed-rank p = 2.37e-49; paired t-test p = 2.78e-05 (t = -4.22)
- excluding |rel gap| > 1%: mean rel gap +0.0052%, Wilcoxon p = 3.42e-67

### track v̄=1.5  [jaxipm solved 623/1000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 623)

- mean objective: jaxipm 546.889, IPOPT 558.628; median jaxipm 540.397, IPOPT 544.146
- mean paired difference -11.7392 (std 75.9231); median -0.00000; mean relative gap -1.178% (median -0.0000%)
- 95% bootstrap CI of the mean relative gap: [-1.959%, -0.318%]
- jaxipm lower by >0.0001 rel: 13.6%; equal within 0.0001: 74.2%; jaxipm higher by >0.0001: 12.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.036%, -40.246%, -15.453%, -0.000%, +0.660%, +2.781%, +157.042%
- instances with |rel gap| > 1% (different local optimum or failed solve): 67 (10.75%)
- Wilcoxon signed-rank p = 1.08e-27; paired t-test p = 0.000126 (t = -3.86)
- excluding |rel gap| > 1%: mean rel gap +0.0101%, Wilcoxon p = 3.85e-29

### track v̄=2.0  [jaxipm solved 631/1000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 631)

- mean objective: jaxipm 1155.340, IPOPT 1151.595; median jaxipm 1148.809, IPOPT 1148.029
- mean paired difference +3.7452 (std 82.2610); median +0.00000; mean relative gap +0.433% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.114%, +1.137%]
- jaxipm lower by >0.0001 rel: 25.0%; equal within 0.0001: 54.8%; jaxipm higher by >0.0001: 20.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -25.403%, -13.891%, -3.037%, +0.000%, +5.008%, +14.414%, +103.944%
- instances with |rel gap| > 1% (different local optimum or failed solve): 170 (26.94%)
- Wilcoxon signed-rank p = 0.0732; paired t-test p = 0.253 (t = +1.14)
- excluding |rel gap| > 1%: mean rel gap -0.0131%, Wilcoxon p = 2.43e-05

### multi N=2: one fixed problem solved 75× (jaxipm) / 75× (IPOPT)

- IPOPT objective 756.3109 (std 0.00e+00); jaxipm median 756.3109, mean 753.4267, min 540.0000, max 756.3109; jaxipm within 1e-4 rel of IPOPT: 98.7%; jaxipm success flags 74/75

### multi N=4: one fixed problem solved 75× (jaxipm) / 75× (IPOPT)

- IPOPT objective 1537.7159 (std 4.55e-13); jaxipm median 1537.7159, mean 1531.6012, min 1080.0000, max 1537.7159; jaxipm within 1e-4 rel of IPOPT: 98.7%; jaxipm success flags 74/75
