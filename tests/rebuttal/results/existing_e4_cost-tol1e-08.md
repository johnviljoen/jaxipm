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

### nav 90° ablation hr_ir0_b500_dbg_iclog  [1663 distinct instances]: paired final objective, jaxipm − IPOPT (n = 1663)

- mean objective: jaxipm 517.147, IPOPT 517.948; median jaxipm 510.561, IPOPT 511.456
- mean paired difference -0.8013 (std 0.4686); median -0.68547; mean relative gap -0.154% (median -0.1349%)
- 95% bootstrap CI of the mean relative gap: [-0.158%, -0.149%]
- jaxipm lower by >0.0001 rel: 99.9%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.369%, -0.306%, -0.287%, -0.135%, -0.101%, -0.093%, +1.887%
- instances with |rel gap| > 1% (different local optimum or failed solve): 2 (0.12%)
- Wilcoxon signed-rank p = 1.08e-270; paired t-test p = 0 (t = -69.74)
- excluding |rel gap| > 1%: mean rel gap -0.1562%, Wilcoxon p = 5.77e-273

### nav 90° ablation hr_ir0_b500_dbg  [1663 distinct instances]: paired final objective, jaxipm − IPOPT (n = 1663)

- mean objective: jaxipm 517.009, IPOPT 517.817; median jaxipm 510.528, IPOPT 511.408
- mean paired difference -0.8072 (std 0.3891); median -0.68484; mean relative gap -0.155% (median -0.1349%)
- 95% bootstrap CI of the mean relative gap: [-0.158%, -0.151%]
- jaxipm lower by >0.0001 rel: 99.9%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.369%, -0.306%, -0.287%, -0.135%, -0.100%, -0.093%, +1.655%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (0.06%)
- Wilcoxon signed-rank p = 5.46e-272; paired t-test p = 0 (t = -84.61)
- excluding |rel gap| > 1%: mean rel gap -0.1561%, Wilcoxon p = 3.96e-273

### nav 90° ablation hr_ir0_b8_dbg_smoke4  [24 distinct instances]: paired final objective, jaxipm − IPOPT (n = 24)

- mean objective: jaxipm 497.783, IPOPT 498.304; median jaxipm 497.784, IPOPT 498.304
- mean paired difference -0.5206 (std 0.0001); median -0.52053; mean relative gap -0.104% (median -0.1045%)
- 95% bootstrap CI of the mean relative gap: [-0.104%, -0.104%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.105%, -0.105%, -0.105%, -0.104%, -0.104%, -0.104%, -0.104%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 1.19e-07; paired t-test p = 2.6e-86 (t = -23327.73)

### nav 90° ablation ws_ir0_b1_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 515.852, IPOPT 516.634; median jaxipm 509.772, IPOPT 510.458
- mean paired difference -0.7817 (std 0.2965); median -0.67336; mean relative gap -0.151% (median -0.1324%)
- 95% bootstrap CI of the mean relative gap: [-0.166%, -0.136%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.295%, -0.291%, -0.268%, -0.132%, -0.100%, -0.094%, -0.094%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 1.63e-09; paired t-test p = 5.53e-23 (t = -18.27)

### nav 90° ablation ws_ir0_b1_dbg_iclog  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 515.845, IPOPT 516.634; median jaxipm 509.772, IPOPT 510.458
- mean paired difference -0.7883 (std 0.2999); median -0.67336; mean relative gap -0.152% (median -0.1324%)
- 95% bootstrap CI of the mean relative gap: [-0.168%, -0.137%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.295%, -0.291%, -0.268%, -0.132%, -0.101%, -0.094%, -0.094%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 7.11e-15; paired t-test p = 6.26e-23 (t = -18.21)

### nav 90° ablation ws_ir0_b32_dbg_i2  [63 distinct instances]: paired final objective, jaxipm − IPOPT (n = 63)

- mean objective: jaxipm 516.343, IPOPT 516.829; median jaxipm 510.637, IPOPT 511.456
- mean paired difference -0.4868 (std 1.7420); median -0.68443; mean relative gap -0.094% (median -0.1297%)
- 95% bootstrap CI of the mean relative gap: [-0.159%, -0.001%]
- jaxipm lower by >0.0001 rel: 96.8%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 3.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.291%, -0.291%, -0.270%, -0.130%, -0.097%, +1.715%, +1.727%
- instances with |rel gap| > 1% (different local optimum or failed solve): 2 (3.17%)
- Wilcoxon signed-rank p = 1.49e-09; paired t-test p = 0.0302 (t = -2.22)
- excluding |rel gap| > 1%: mean rel gap -0.1532%, Wilcoxon p = 1.11e-11

### nav 90° ablation ws_ir0_b32_dbg_iclog  [96 distinct instances]: paired final objective, jaxipm − IPOPT (n = 96)

- mean objective: jaxipm 515.997, IPOPT 516.801; median jaxipm 509.822, IPOPT 510.519
- mean paired difference -0.8042 (std 0.3125); median -0.68234; mean relative gap -0.155% (median -0.1348%)
- 95% bootstrap CI of the mean relative gap: [-0.167%, -0.144%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.310%, -0.306%, -0.282%, -0.135%, -0.100%, -0.093%, -0.093%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 1.78e-17; paired t-test p = 7.12e-44 (t = -25.22)

### nav 90° ablation ws_ir0_b500_dbg  [2000 distinct instances]: paired final objective, jaxipm − IPOPT (n = 2000)

- mean objective: jaxipm 516.164, IPOPT 516.953; median jaxipm 510.093, IPOPT 510.968
- mean paired difference -0.7896 (std 0.4898); median -0.68482; mean relative gap -0.152% (median -0.1348%)
- 95% bootstrap CI of the mean relative gap: [-0.156%, -0.148%]
- jaxipm lower by >0.0001 rel: 99.9%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.369%, -0.306%, -0.280%, -0.135%, -0.099%, -0.093%, +1.888%
- instances with |rel gap| > 1% (different local optimum or failed solve): 3 (0.15%)
- Wilcoxon signed-rank p = 0; paired t-test p = 0 (t = -72.09)
- excluding |rel gap| > 1%: mean rel gap -0.1548%, Wilcoxon p = 0

### nav 90° ablation ws_ir0_b8_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 515.829, IPOPT 516.634; median jaxipm 509.772, IPOPT 510.458
- mean paired difference -0.8047 (std 0.3222); median -0.67336; mean relative gap -0.155% (median -0.1324%)
- 95% bootstrap CI of the mean relative gap: [-0.172%, -0.139%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.308%, -0.302%, -0.282%, -0.132%, -0.101%, -0.094%, -0.094%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 7.11e-15; paired t-test p = 5.1e-22 (t = -17.30)

### nav 90° ablation ws_ir0_b8_dbg_smoke5  [16 distinct instances]: paired final objective, jaxipm − IPOPT (n = 16)

- mean objective: jaxipm 514.445, IPOPT 515.245; median jaxipm 510.291, IPOPT 510.963
- mean paired difference -0.8003 (std 0.3251); median -0.64979; mean relative gap -0.154% (median -0.1290%)
- 95% bootstrap CI of the mean relative gap: [-0.183%, -0.128%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.288%, -0.283%, -0.265%, -0.129%, -0.104%, -0.104%, -0.104%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 3.05e-05; paired t-test p = 6.13e-08 (t = -9.85)

### nav 90° ablation ws_ir0_b8_dbg_smoke6  [16 distinct instances]: paired final objective, jaxipm − IPOPT (n = 16)

- mean objective: jaxipm 514.425, IPOPT 515.245; median jaxipm 510.291, IPOPT 510.963
- mean paired difference -0.8205 (std 0.3216); median -0.70619; mean relative gap -0.158% (median -0.1398%)
- 95% bootstrap CI of the mean relative gap: [-0.186%, -0.132%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.288%, -0.283%, -0.265%, -0.140%, -0.104%, -0.104%, -0.104%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 3.05e-05; paired t-test p = 3.82e-08 (t = -10.21)

### nav 90° ablation ws_ir100_b1_dbg_i2  [47 distinct instances]: paired final objective, jaxipm − IPOPT (n = 47)

- mean objective: jaxipm 515.597, IPOPT 516.411; median jaxipm 509.751, IPOPT 510.448
- mean paired difference -0.8143 (std 0.3271); median -0.67697; mean relative gap -0.157% (median -0.1326%)
- 95% bootstrap CI of the mean relative gap: [-0.175%, -0.140%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.308%, -0.302%, -0.282%, -0.133%, -0.102%, -0.097%, -0.094%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 1.42e-14; paired t-test p = 1.57e-21 (t = -17.07)

### nav 90° ablation ws_ir100_b32_dbg_i2  [61 distinct instances]: paired final objective, jaxipm − IPOPT (n = 61)

- mean objective: jaxipm 513.926, IPOPT 514.713; median jaxipm 509.268, IPOPT 509.908
- mean paired difference -0.7869 (std 0.3063); median -0.66074; mean relative gap -0.152% (median -0.1297%)
- 95% bootstrap CI of the mean relative gap: [-0.167%, -0.139%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.321%, -0.303%, -0.270%, -0.130%, -0.102%, -0.096%, -0.095%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 1.11e-11; paired t-test p = 2.68e-28 (t = -20.07)

### nav 90° ablation ws_ir100_b8_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 515.831, IPOPT 516.634; median jaxipm 509.772, IPOPT 510.458
- mean paired difference -0.8024 (std 0.3185); median -0.67336; mean relative gap -0.154% (median -0.1324%)
- 95% bootstrap CI of the mean relative gap: [-0.171%, -0.139%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.328%, -0.309%, -0.268%, -0.132%, -0.101%, -0.094%, -0.094%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 7.11e-15; paired t-test p = 3.58e-22 (t = -17.45)

### nav 180°  [jaxipm solved 1634/2000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 1634)

- mean objective: jaxipm 507.009, IPOPT 507.803; median jaxipm 497.796, IPOPT 498.305
- mean paired difference -0.7941 (std 6.7160); median -0.66834; mean relative gap -0.157% (median -0.1326%)
- 95% bootstrap CI of the mean relative gap: [-0.230%, -0.107%]
- jaxipm lower by >0.0001 rel: 99.4%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.6%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.052%, -0.306%, -0.266%, -0.133%, -0.099%, -0.093%, +13.895%
- instances with |rel gap| > 1% (different local optimum or failed solve): 10 (0.61%)
- Wilcoxon signed-rank p = 5.36e-257; paired t-test p = 1.91e-06 (t = -4.78)
- excluding |rel gap| > 1%: mean rel gap -0.1521%, Wilcoxon p = 1.24e-265

### nav 180° ablation hr_ir0_b500_dbg  [1574 distinct instances]: paired final objective, jaxipm − IPOPT (n = 1574)

- mean objective: jaxipm 507.416, IPOPT 508.189; median jaxipm 497.782, IPOPT 498.305
- mean paired difference -0.7727 (std 0.5131); median -0.70155; mean relative gap -0.152% (median -0.1396%)
- 95% bootstrap CI of the mean relative gap: [-0.156%, -0.146%]
- jaxipm lower by >0.0001 rel: 99.8%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.381%, -0.305%, -0.264%, -0.140%, -0.104%, -0.094%, +1.914%
- instances with |rel gap| > 1% (different local optimum or failed solve): 3 (0.19%)
- Wilcoxon signed-rank p = 6.84e-255; paired t-test p = 0 (t = -59.75)
- excluding |rel gap| > 1%: mean rel gap -0.1553%, Wilcoxon p = 2.7e-258

### nav 180° ablation ws_ir0_b1_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 506.536, IPOPT 507.312; median jaxipm 497.779, IPOPT 498.304
- mean paired difference -0.7759 (std 0.2531); median -0.69020; mean relative gap -0.153% (median -0.1362%)
- 95% bootstrap CI of the mean relative gap: [-0.167%, -0.140%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.306%, -0.284%, -0.225%, -0.136%, -0.105%, -0.097%, -0.097%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 7.11e-15; paired t-test p = 9.79e-26 (t = -21.24)

### nav 180° ablation ws_ir0_b32_dbg_i2  [64 distinct instances]: paired final objective, jaxipm − IPOPT (n = 64)

- mean objective: jaxipm 506.516, IPOPT 507.309; median jaxipm 497.781, IPOPT 498.304
- mean paired difference -0.7929 (std 0.2797); median -0.71082; mean relative gap -0.156% (median -0.1403%)
- 95% bootstrap CI of the mean relative gap: [-0.169%, -0.144%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.308%, -0.306%, -0.259%, -0.140%, -0.105%, -0.099%, -0.099%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 3.53e-12; paired t-test p = 5.27e-32 (t = -22.68)

### nav 180° ablation ws_ir0_b500_dbg  [1999 distinct instances]: paired final objective, jaxipm − IPOPT (n = 1999)

- mean objective: jaxipm 506.816, IPOPT 507.576; median jaxipm 497.781, IPOPT 498.304
- mean paired difference -0.7598 (std 0.5243); median -0.68932; mean relative gap -0.149% (median -0.1370%)
- 95% bootstrap CI of the mean relative gap: [-0.153%, -0.145%]
- jaxipm lower by >0.0001 rel: 99.8%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.381%, -0.299%, -0.259%, -0.137%, -0.104%, -0.094%, +1.914%
- instances with |rel gap| > 1% (different local optimum or failed solve): 4 (0.20%)
- Wilcoxon signed-rank p = 0; paired t-test p = 0 (t = -64.79)
- excluding |rel gap| > 1%: mean rel gap -0.1533%, Wilcoxon p = 0

### nav 180° ablation ws_ir0_b8_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 506.536, IPOPT 507.312; median jaxipm 497.779, IPOPT 498.304
- mean paired difference -0.7759 (std 0.2531); median -0.69020; mean relative gap -0.153% (median -0.1362%)
- 95% bootstrap CI of the mean relative gap: [-0.167%, -0.140%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.306%, -0.284%, -0.225%, -0.136%, -0.105%, -0.097%, -0.097%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 7.11e-15; paired t-test p = 9.79e-26 (t = -21.24)

### nav 180° ablation ws_ir100_b1_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 506.548, IPOPT 507.312; median jaxipm 497.779, IPOPT 498.304
- mean paired difference -0.7636 (std 0.2611); median -0.67657; mean relative gap -0.151% (median -0.1349%)
- 95% bootstrap CI of the mean relative gap: [-0.165%, -0.137%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.306%, -0.284%, -0.225%, -0.135%, -0.102%, -0.073%, -0.053%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 7.11e-15; paired t-test p = 7.24e-25 (t = -20.26)

### nav 180° ablation ws_ir100_b32_dbg_i2  [60 distinct instances]: paired final objective, jaxipm − IPOPT (n = 60)

- mean objective: jaxipm 506.182, IPOPT 506.613; median jaxipm 497.782, IPOPT 498.304
- mean paired difference -0.4304 (std 2.8280); median -0.67243; mean relative gap -0.086% (median -0.1342%)
- 95% bootstrap CI of the mean relative gap: [-0.167%, +0.063%]
- jaxipm lower by >0.0001 rel: 98.3%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 1.7%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.349%, -0.325%, -0.267%, -0.134%, -0.099%, +1.609%, +4.067%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (1.67%)
- Wilcoxon signed-rank p = 3.09e-10; paired t-test p = 0.243 (t = -1.18)
- excluding |rel gap| > 1%: mean rel gap -0.1562%, Wilcoxon p = 2.39e-11

### nav 180° ablation ws_ir100_b8_dbg_i2  [46 distinct instances]: paired final objective, jaxipm − IPOPT (n = 46)

- mean objective: jaxipm 506.968, IPOPT 507.720; median jaxipm 497.781, IPOPT 498.304
- mean paired difference -0.7517 (std 0.2545); median -0.67258; mean relative gap -0.148% (median -0.1335%)
- 95% bootstrap CI of the mean relative gap: [-0.162%, -0.134%]
- jaxipm lower by >0.0001 rel: 100.0%; equal within 0.0001: 0.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.306%, -0.285%, -0.234%, -0.133%, -0.101%, -0.085%, -0.076%
- instances with |rel gap| > 1% (different local optimum or failed solve): 0 (0.00%)
- Wilcoxon signed-rank p = 2.84e-14; paired t-test p = 4.69e-24 (t = -20.04)

### track v̄=1.0  [jaxipm solved 628/1000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 628)

- mean objective: jaxipm 254.284, IPOPT 268.011; median jaxipm 241.287, IPOPT 247.100
- mean paired difference -13.7272 (std 81.4833); median -0.00000; mean relative gap -2.161% (median -0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.803%, -0.135%]
- jaxipm lower by >0.0001 rel: 12.3%; equal within 0.0001: 78.3%; jaxipm higher by >0.0001: 9.4%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -73.573%, -62.925%, -32.468%, -0.000%, +1.359%, +20.071%, +359.623%
- instances with |rel gap| > 1% (different local optimum or failed solve): 111 (17.68%)
- Wilcoxon signed-rank p = 0.388; paired t-test p = 2.78e-05 (t = -4.22)
- excluding |rel gap| > 1%: mean rel gap +0.0052%, Wilcoxon p = 0.000316

### track v̄=1.0 ablation hr_ir0_b250_dbg  [634 distinct instances]: paired final objective, jaxipm − IPOPT (n = 634)

- mean objective: jaxipm 246.264, IPOPT 263.589; median jaxipm 241.389, IPOPT 245.766
- mean paired difference -17.3250 (std 73.4699); median +0.00000; mean relative gap -3.075% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-4.380%, -1.674%]
- jaxipm lower by >0.0001 rel: 8.5%; equal within 0.0001: 82.8%; jaxipm higher by >0.0001: 8.7%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -69.193%, -63.796%, -50.251%, +0.000%, +1.252%, +1.795%, +265.211%
- instances with |rel gap| > 1% (different local optimum or failed solve): 86 (13.56%)
- Wilcoxon signed-rank p = 6.07e-44; paired t-test p = 4.77e-09 (t = -5.94)
- excluding |rel gap| > 1%: mean rel gap +0.0072%, Wilcoxon p = 1.85e-72

### track v̄=1.0 ablation ws_ir0_b1_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 250.893, IPOPT 247.317; median jaxipm 242.335, IPOPT 242.335
- mean paired difference +3.5767 (std 70.8648); median +0.00000; mean relative gap +4.414% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.504%, +16.692%]
- jaxipm lower by >0.0001 rel: 4.2%; equal within 0.0001: 83.3%; jaxipm higher by >0.0001: 12.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -58.671%, -31.195%, -0.000%, +0.000%, +1.354%, +141.217%, +265.211%
- instances with |rel gap| > 1% (different local optimum or failed solve): 6 (12.50%)
- Wilcoxon signed-rank p = 8.81e-08; paired t-test p = 0.728 (t = +0.35)
- excluding |rel gap| > 1%: mean rel gap +0.0012%, Wilcoxon p = 7.11e-08

### track v̄=1.0 ablation ws_ir0_b250_dbg  [993 distinct instances]: paired final objective, jaxipm − IPOPT (n = 993)

- mean objective: jaxipm 249.396, IPOPT 265.863; median jaxipm 241.456, IPOPT 246.636
- mean paired difference -16.4672 (std 72.8770); median +0.00000; mean relative gap -2.970% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.960%, -1.822%]
- jaxipm lower by >0.0001 rel: 8.3%; equal within 0.0001: 82.6%; jaxipm higher by >0.0001: 9.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -73.573%, -65.260%, -49.058%, +0.000%, +1.180%, +1.676%, +265.211%
- instances with |rel gap| > 1% (different local optimum or failed solve): 130 (13.09%)
- Wilcoxon signed-rank p = 2.49e-69; paired t-test p = 2.06e-12 (t = -7.12)
- excluding |rel gap| > 1%: mean rel gap +0.0127%, Wilcoxon p = 1.82e-114

### track v̄=1.0 ablation ws_ir0_b32_dbg_i2  [64 distinct instances]: paired final objective, jaxipm − IPOPT (n = 64)

- mean objective: jaxipm 270.587, IPOPT 277.586; median jaxipm 246.627, IPOPT 248.140
- mean paired difference -6.9991 (std 43.9293); median +0.00000; mean relative gap -1.400% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.812%, +0.142%]
- jaxipm lower by >0.0001 rel: 6.2%; equal within 0.0001: 81.2%; jaxipm higher by >0.0001: 12.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -62.923%, -34.641%, -0.035%, +0.000%, +1.307%, +1.582%, +1.588%
- instances with |rel gap| > 1% (different local optimum or failed solve): 9 (14.06%)
- Wilcoxon signed-rank p = 4.6e-07; paired t-test p = 0.207 (t = -1.27)
- excluding |rel gap| > 1%: mean rel gap +0.0034%, Wilcoxon p = 1.71e-08

### track v̄=1.0 ablation ws_ir0_b8_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 250.893, IPOPT 247.317; median jaxipm 242.335, IPOPT 242.335
- mean paired difference +3.5767 (std 70.8648); median +0.00000; mean relative gap +4.414% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.504%, +16.692%]
- jaxipm lower by >0.0001 rel: 4.2%; equal within 0.0001: 83.3%; jaxipm higher by >0.0001: 12.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -58.671%, -31.195%, -0.000%, +0.000%, +1.354%, +141.217%, +265.211%
- instances with |rel gap| > 1% (different local optimum or failed solve): 6 (12.50%)
- Wilcoxon signed-rank p = 8.81e-08; paired t-test p = 0.728 (t = +0.35)
- excluding |rel gap| > 1%: mean rel gap +0.0012%, Wilcoxon p = 7.11e-08

### track v̄=1.0 ablation ws_ir100_b1_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 250.893, IPOPT 247.317; median jaxipm 242.335, IPOPT 242.335
- mean paired difference +3.5767 (std 70.8648); median +0.00000; mean relative gap +4.414% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.504%, +16.692%]
- jaxipm lower by >0.0001 rel: 4.2%; equal within 0.0001: 83.3%; jaxipm higher by >0.0001: 12.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -58.671%, -31.195%, -0.000%, +0.000%, +1.354%, +141.217%, +265.211%
- instances with |rel gap| > 1% (different local optimum or failed solve): 6 (12.50%)
- Wilcoxon signed-rank p = 8.81e-08; paired t-test p = 0.728 (t = +0.35)
- excluding |rel gap| > 1%: mean rel gap +0.0012%, Wilcoxon p = 7.11e-08

### track v̄=1.0 ablation ws_ir100_b32_dbg_i2  [63 distinct instances]: paired final objective, jaxipm − IPOPT (n = 63)

- mean objective: jaxipm 264.478, IPOPT 276.049; median jaxipm 246.385, IPOPT 246.385
- mean paired difference -11.5705 (std 48.8757); median +0.00000; mean relative gap -2.484% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-5.218%, -0.483%]
- jaxipm lower by >0.0001 rel: 11.1%; equal within 0.0001: 76.2%; jaxipm higher by >0.0001: 12.7%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -62.923%, -46.974%, -17.936%, +0.000%, +1.185%, +1.461%, +1.578%
- instances with |rel gap| > 1% (different local optimum or failed solve): 11 (17.46%)
- Wilcoxon signed-rank p = 8.51e-05; paired t-test p = 0.0649 (t = -1.88)
- excluding |rel gap| > 1%: mean rel gap +0.0091%, Wilcoxon p = 1.82e-08

### track v̄=1.0 ablation ws_ir100_b8_dbg_i2  [48 distinct instances]: paired final objective, jaxipm − IPOPT (n = 48)

- mean objective: jaxipm 250.895, IPOPT 247.317; median jaxipm 242.335, IPOPT 242.335
- mean paired difference +3.5785 (std 70.8581); median +0.00000; mean relative gap +4.414% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.503%, +16.692%]
- jaxipm lower by >0.0001 rel: 4.2%; equal within 0.0001: 83.3%; jaxipm higher by >0.0001: 12.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -58.651%, -31.185%, -0.000%, +0.000%, +1.354%, +141.217%, +265.211%
- instances with |rel gap| > 1% (different local optimum or failed solve): 6 (12.50%)
- Wilcoxon signed-rank p = 8.81e-08; paired t-test p = 0.728 (t = +0.35)
- excluding |rel gap| > 1%: mean rel gap +0.0012%, Wilcoxon p = 7.11e-08

### track v̄=1.5  [jaxipm solved 623/1000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 623)

- mean objective: jaxipm 546.889, IPOPT 558.628; median jaxipm 540.397, IPOPT 544.146
- mean paired difference -11.7392 (std 75.9231); median +0.00000; mean relative gap -1.178% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-1.959%, -0.318%]
- jaxipm lower by >0.0001 rel: 13.6%; equal within 0.0001: 74.2%; jaxipm higher by >0.0001: 12.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.036%, -40.246%, -15.453%, +0.000%, +0.660%, +2.781%, +157.042%
- instances with |rel gap| > 1% (different local optimum or failed solve): 67 (10.75%)
- Wilcoxon signed-rank p = 5.03e-15; paired t-test p = 0.000126 (t = -3.86)
- excluding |rel gap| > 1%: mean rel gap +0.0101%, Wilcoxon p = 3.45e-28

### track v̄=1.5 ablation hr_ir0_b250_dbg  [652 distinct instances]: paired final objective, jaxipm − IPOPT (n = 652)

- mean objective: jaxipm 546.871, IPOPT 560.302; median jaxipm 541.385, IPOPT 547.315
- mean paired difference -13.4306 (std 77.2450); median +0.00000; mean relative gap -1.471% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-2.148%, -0.788%]
- jaxipm lower by >0.0001 rel: 11.3%; equal within 0.0001: 77.8%; jaxipm higher by >0.0001: 10.9%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.399%, -42.481%, -18.277%, +0.000%, +0.703%, +2.122%, +73.658%
- instances with |rel gap| > 1% (different local optimum or failed solve): 64 (9.82%)
- Wilcoxon signed-rank p = 5e-38; paired t-test p = 1.06e-05 (t = -4.44)
- excluding |rel gap| > 1%: mean rel gap +0.0145%, Wilcoxon p = 5.4e-60

### track v̄=1.5 ablation ws_ir0_b1_dbg_i2  [44 distinct instances]: paired final objective, jaxipm − IPOPT (n = 44)

- mean objective: jaxipm 526.825, IPOPT 537.502; median jaxipm 524.573, IPOPT 535.091
- mean paired difference -10.6771 (std 72.4862); median +0.00000; mean relative gap -1.097% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.397%, +0.077%]
- jaxipm lower by >0.0001 rel: 4.5%; equal within 0.0001: 81.8%; jaxipm higher by >0.0001: 13.6%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.036%, -28.706%, -0.000%, +0.000%, +0.481%, +0.512%, +0.517%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (2.27%)
- Wilcoxon signed-rank p = 1.45e-07; paired t-test p = 0.334 (t = -0.98)
- excluding |rel gap| > 1%: mean rel gap +0.0410%, Wilcoxon p = 2.26e-09

### track v̄=1.5 ablation ws_ir0_b250_dbg  [968 distinct instances]: paired final objective, jaxipm − IPOPT (n = 968)

- mean objective: jaxipm 543.564, IPOPT 556.197; median jaxipm 540.497, IPOPT 544.290
- mean paired difference -12.6328 (std 65.2193); median +0.00000; mean relative gap -1.425% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-1.930%, -0.981%]
- jaxipm lower by >0.0001 rel: 11.6%; equal within 0.0001: 76.2%; jaxipm higher by >0.0001: 12.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.399%, -41.987%, -1.292%, +0.000%, +0.666%, +2.204%, +21.404%
- instances with |rel gap| > 1% (different local optimum or failed solve): 92 (9.50%)
- Wilcoxon signed-rank p = 4.74e-55; paired t-test p = 2.38e-09 (t = -6.03)
- excluding |rel gap| > 1%: mean rel gap +0.0099%, Wilcoxon p = 1.61e-82

### track v̄=1.5 ablation ws_ir0_b32_dbg_i2  [63 distinct instances]: paired final objective, jaxipm − IPOPT (n = 63)

- mean objective: jaxipm 579.838, IPOPT 577.884; median jaxipm 560.479, IPOPT 550.307
- mean paired difference +1.9545 (std 13.7913); median +0.00000; mean relative gap +0.360% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.039%, +1.082%]
- jaxipm lower by >0.0001 rel: 11.1%; equal within 0.0001: 76.2%; jaxipm higher by >0.0001: 12.7%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -0.881%, -0.876%, -0.464%, +0.000%, +0.657%, +9.170%, +20.465%
- instances with |rel gap| > 1% (different local optimum or failed solve): 3 (4.76%)
- Wilcoxon signed-rank p = 2.26e-05; paired t-test p = 0.265 (t = +1.12)
- excluding |rel gap| > 1%: mean rel gap -0.0285%, Wilcoxon p = 0.000108

### track v̄=1.5 ablation ws_ir0_b8_dbg_i2  [44 distinct instances]: paired final objective, jaxipm − IPOPT (n = 44)

- mean objective: jaxipm 525.908, IPOPT 537.502; median jaxipm 524.573, IPOPT 535.091
- mean paired difference -11.5946 (std 72.6611); median +0.00000; mean relative gap -1.236% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.671%, +0.080%]
- jaxipm lower by >0.0001 rel: 6.8%; equal within 0.0001: 77.3%; jaxipm higher by >0.0001: 15.9%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.036%, -31.461%, -0.066%, +0.000%, +0.504%, +0.553%, +0.580%
- instances with |rel gap| > 1% (different local optimum or failed solve): 2 (4.55%)
- Wilcoxon signed-rank p = 3.86e-06; paired t-test p = 0.296 (t = -1.06)
- excluding |rel gap| > 1%: mean rel gap +0.0592%, Wilcoxon p = 3e-09

### track v̄=1.5 ablation ws_ir100_b1_dbg_i2  [46 distinct instances]: paired final objective, jaxipm − IPOPT (n = 46)

- mean objective: jaxipm 518.639, IPOPT 529.005; median jaxipm 515.953, IPOPT 524.573
- mean paired difference -10.3664 (std 70.8655); median +0.00000; mean relative gap -1.074% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.274%, +0.043%]
- jaxipm lower by >0.0001 rel: 6.5%; equal within 0.0001: 87.0%; jaxipm higher by >0.0001: 6.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.036%, -27.714%, -0.055%, +0.000%, +0.114%, +0.496%, +0.517%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (2.17%)
- Wilcoxon signed-rank p = 1.68e-06; paired t-test p = 0.326 (t = -0.99)
- excluding |rel gap| > 1%: mean rel gap +0.0141%, Wilcoxon p = 8.01e-08

### track v̄=1.5 ablation ws_ir100_b32_dbg_i2  [63 distinct instances]: paired final objective, jaxipm − IPOPT (n = 63)

- mean objective: jaxipm 571.867, IPOPT 569.756; median jaxipm 550.307, IPOPT 550.215
- mean paired difference +2.1103 (std 13.9234); median +0.00000; mean relative gap +0.385% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.029%, +1.121%]
- jaxipm lower by >0.0001 rel: 11.1%; equal within 0.0001: 74.6%; jaxipm higher by >0.0001: 14.3%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -1.078%, -0.956%, -0.503%, +0.000%, +1.591%, +9.170%, +20.465%
- instances with |rel gap| > 1% (different local optimum or failed solve): 5 (7.94%)
- Wilcoxon signed-rank p = 2e-05; paired t-test p = 0.234 (t = +1.20)
- excluding |rel gap| > 1%: mean rel gap -0.0172%, Wilcoxon p = 4.14e-05

### track v̄=1.5 ablation ws_ir100_b8_dbg_i2  [46 distinct instances]: paired final objective, jaxipm − IPOPT (n = 46)

- mean objective: jaxipm 521.106, IPOPT 537.224; median jaxipm 524.573, IPOPT 535.091
- mean paired difference -16.1177 (std 80.1349); median +0.00000; mean relative gap -1.786% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-4.677%, +0.029%]
- jaxipm lower by >0.0001 rel: 8.7%; equal within 0.0001: 84.8%; jaxipm higher by >0.0001: 6.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -50.036%, -42.260%, -0.343%, +0.000%, +0.114%, +0.496%, +0.517%
- instances with |rel gap| > 1% (different local optimum or failed solve): 2 (4.35%)
- Wilcoxon signed-rank p = 2.7e-05; paired t-test p = 0.179 (t = -1.36)
- excluding |rel gap| > 1%: mean rel gap +0.0144%, Wilcoxon p = 1.32e-07

### track v̄=2.0  [jaxipm solved 631/1000 distinct pool instances]: paired final objective, jaxipm − IPOPT (n = 631)

- mean objective: jaxipm 1155.340, IPOPT 1151.595; median jaxipm 1148.809, IPOPT 1148.029
- mean paired difference +3.7452 (std 82.2610); median +0.00000; mean relative gap +0.433% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.114%, +1.137%]
- jaxipm lower by >0.0001 rel: 25.0%; equal within 0.0001: 54.8%; jaxipm higher by >0.0001: 20.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -25.403%, -13.891%, -3.037%, +0.000%, +5.008%, +14.414%, +103.944%
- instances with |rel gap| > 1% (different local optimum or failed solve): 170 (26.94%)
- Wilcoxon signed-rank p = 7.08e-07; paired t-test p = 0.253 (t = +1.14)
- excluding |rel gap| > 1%: mean rel gap -0.0131%, Wilcoxon p = 1.53e-20

### track v̄=2.0 ablation hr_ir0_b250_dbg  [668 distinct instances]: paired final objective, jaxipm − IPOPT (n = 668)

- mean objective: jaxipm 1149.196, IPOPT 1149.709; median jaxipm 1147.128, IPOPT 1145.850
- mean paired difference -0.5127 (std 64.3787); median +0.00000; mean relative gap +0.039% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.343%, +0.494%]
- jaxipm lower by >0.0001 rel: 23.4%; equal within 0.0001: 54.6%; jaxipm higher by >0.0001: 22.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -33.091%, -14.319%, -3.578%, +0.000%, +5.012%, +14.229%, +105.216%
- instances with |rel gap| > 1% (different local optimum or failed solve): 186 (27.84%)
- Wilcoxon signed-rank p = 1.14e-09; paired t-test p = 0.837 (t = -0.21)
- excluding |rel gap| > 1%: mean rel gap +0.0136%, Wilcoxon p = 1.98e-29

### track v̄=2.0 ablation ws_ir0_b1_dbg_i2  [10 distinct instances]: paired final objective, jaxipm − IPOPT (n = 10)

- mean objective: jaxipm 1088.855, IPOPT 1094.715; median jaxipm 1074.441, IPOPT 1094.600
- mean paired difference -5.8604 (std 17.6376); median +0.00001; mean relative gap -0.544% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-1.609%, +0.000%]
- jaxipm lower by >0.0001 rel: 20.0%; equal within 0.0001: 80.0%; jaxipm higher by >0.0001: 0.0%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -5.209%, -4.761%, -2.970%, +0.000%, +0.000%, +0.000%, +0.000%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (10.00%)
- Wilcoxon signed-rank p = 0.432; paired t-test p = 0.321 (t = -1.05)
- excluding |rel gap| > 1%: mean rel gap -0.0259%, Wilcoxon p = 0.129

### track v̄=2.0 ablation ws_ir0_b250_dbg  [858 distinct instances]: paired final objective, jaxipm − IPOPT (n = 858)

- mean objective: jaxipm 1142.525, IPOPT 1144.471; median jaxipm 1140.998, IPOPT 1140.653
- mean paired difference -1.9463 (std 52.3347); median +0.00000; mean relative gap -0.108% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.386%, +0.164%]
- jaxipm lower by >0.0001 rel: 22.7%; equal within 0.0001: 58.7%; jaxipm higher by >0.0001: 18.5%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -33.091%, -14.137%, -3.834%, +0.000%, +4.636%, +14.238%, +48.115%
- instances with |rel gap| > 1% (different local optimum or failed solve): 227 (26.46%)
- Wilcoxon signed-rank p = 5.6e-12; paired t-test p = 0.276 (t = -1.09)
- excluding |rel gap| > 1%: mean rel gap -0.0034%, Wilcoxon p = 4.17e-40

### track v̄=2.0 ablation ws_ir0_b250_dbg_sub250  [213 distinct instances]: paired final objective, jaxipm − IPOPT (n = 213)

- mean objective: jaxipm 1145.903, IPOPT 1151.258; median jaxipm 1150.635, IPOPT 1151.901
- mean paired difference -5.3552 (std 58.4201); median +0.00000; mean relative gap -0.353% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.968%, +0.181%]
- jaxipm lower by >0.0001 rel: 19.7%; equal within 0.0001: 61.0%; jaxipm higher by >0.0001: 19.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -33.091%, -23.186%, -3.238%, +0.000%, +4.707%, +6.735%, +14.341%
- instances with |rel gap| > 1% (different local optimum or failed solve): 59 (27.70%)
- Wilcoxon signed-rank p = 7.08e-06; paired t-test p = 0.182 (t = -1.34)
- excluding |rel gap| > 1%: mean rel gap +0.0118%, Wilcoxon p = 3.24e-17

### track v̄=2.0 ablation ws_ir0_b32_dbg_i2  [49 distinct instances]: paired final objective, jaxipm − IPOPT (n = 49)

- mean objective: jaxipm 1139.567, IPOPT 1140.340; median jaxipm 1129.278, IPOPT 1124.816
- mean paired difference -0.7723 (std 31.8659); median +0.00000; mean relative gap -0.030% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.851%, +0.867%]
- jaxipm lower by >0.0001 rel: 20.4%; equal within 0.0001: 59.2%; jaxipm higher by >0.0001: 20.4%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -12.635%, -7.989%, -2.757%, +0.000%, +1.894%, +10.466%, +14.317%
- instances with |rel gap| > 1% (different local optimum or failed solve): 11 (22.45%)
- Wilcoxon signed-rank p = 0.0387; paired t-test p = 0.866 (t = -0.17)
- excluding |rel gap| > 1%: mean rel gap +0.0493%, Wilcoxon p = 2.87e-05

### track v̄=2.0 ablation ws_ir0_b48_dbg_sub48  [42 distinct instances]: paired final objective, jaxipm − IPOPT (n = 42)

- mean objective: jaxipm 1130.316, IPOPT 1148.385; median jaxipm 1141.982, IPOPT 1135.836
- mean paired difference -18.0698 (std 99.6700); median +0.00000; mean relative gap -1.083% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-3.383%, +0.716%]
- jaxipm lower by >0.0001 rel: 26.2%; equal within 0.0001: 61.9%; jaxipm higher by >0.0001: 11.9%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -33.091%, -29.939%, -3.084%, +0.000%, +4.794%, +11.150%, +13.958%
- instances with |rel gap| > 1% (different local optimum or failed solve): 11 (26.19%)
- Wilcoxon signed-rank p = 0.552; paired t-test p = 0.247 (t = -1.17)
- excluding |rel gap| > 1%: mean rel gap +0.0003%, Wilcoxon p = 0.00709

### track v̄=2.0 ablation ws_ir0_b8_dbg_i2  [11 distinct instances]: paired final objective, jaxipm − IPOPT (n = 11)

- mean objective: jaxipm 1090.569, IPOPT 1095.880; median jaxipm 1107.717, IPOPT 1107.531
- mean paired difference -5.3107 (std 16.8315); median +0.00001; mean relative gap -0.493% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-1.460%, +0.003%]
- jaxipm lower by >0.0001 rel: 18.2%; equal within 0.0001: 72.7%; jaxipm higher by >0.0001: 9.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -5.209%, -4.712%, -2.721%, +0.000%, +0.008%, +0.015%, +0.017%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (9.09%)
- Wilcoxon signed-rank p = 0.32; paired t-test p = 0.32 (t = -1.05)
- excluding |rel gap| > 1%: mean rel gap -0.0217%, Wilcoxon p = 0.084

### track v̄=2.0 ablation ws_ir100_b1_dbg_i2  [11 distinct instances]: paired final objective, jaxipm − IPOPT (n = 11)

- mean objective: jaxipm 1083.243, IPOPT 1089.576; median jaxipm 1037.393, IPOPT 1075.113
- mean paired difference -6.3327 (std 17.6288); median +0.00001; mean relative gap -0.600% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-1.642%, +0.086%]
- jaxipm lower by >0.0001 rel: 27.3%; equal within 0.0001: 54.5%; jaxipm higher by >0.0001: 18.2%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -5.209%, -4.873%, -3.531%, +0.000%, +0.345%, +0.426%, +0.446%
- instances with |rel gap| > 1% (different local optimum or failed solve): 2 (18.18%)
- Wilcoxon signed-rank p = 0.7; paired t-test p = 0.261 (t = -1.19)
- excluding |rel gap| > 1%: mean rel gap +0.0507%, Wilcoxon p = 0.0742

### track v̄=2.0 ablation ws_ir100_b32_dbg_i2  [51 distinct instances]: paired final objective, jaxipm − IPOPT (n = 51)

- mean objective: jaxipm 1150.669, IPOPT 1148.686; median jaxipm 1131.640, IPOPT 1128.809
- mean paired difference +1.9827 (std 36.2063); median +0.00000; mean relative gap +0.248% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.661%, +1.268%]
- jaxipm lower by >0.0001 rel: 17.6%; equal within 0.0001: 60.8%; jaxipm higher by >0.0001: 21.6%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -12.635%, -7.796%, -2.743%, +0.000%, +4.313%, +14.052%, +14.317%
- instances with |rel gap| > 1% (different local optimum or failed solve): 11 (21.57%)
- Wilcoxon signed-rank p = 0.00892; paired t-test p = 0.697 (t = +0.39)
- excluding |rel gap| > 1%: mean rel gap +0.0195%, Wilcoxon p = 1.95e-05

### track v̄=2.0 ablation ws_ir100_b8_dbg_i2  [11 distinct instances]: paired final objective, jaxipm − IPOPT (n = 11)

- mean objective: jaxipm 1092.620, IPOPT 1094.127; median jaxipm 1111.488, IPOPT 1114.088
- mean paired difference -1.5075 (std 6.1554); median +0.00000; mean relative gap -0.149% (median +0.0000%)
- 95% bootstrap CI of the mean relative gap: [-0.526%, +0.081%]
- jaxipm lower by >0.0001 rel: 18.2%; equal within 0.0001: 72.7%; jaxipm higher by >0.0001: 9.1%
- relative-gap quantiles (0/1/5/50/95/99/100 %): -1.852%, -1.690%, -1.043%, +0.000%, +0.223%, +0.401%, +0.446%
- instances with |rel gap| > 1% (different local optimum or failed solve): 1 (9.09%)
- Wilcoxon signed-rank p = 0.278; paired t-test p = 0.436 (t = -0.81)
- excluding |rel gap| > 1%: mean rel gap +0.0213%, Wilcoxon p = 0.0645

### multi N=2: one fixed problem solved 75× (jaxipm) / 75× (IPOPT)

- IPOPT objective 756.3109 (std 1.14e-13); jaxipm median 756.3109, mean 753.4267, min 540.0000, max 756.3109; jaxipm within 1e-4 rel of IPOPT: 98.7%; jaxipm success flags 74/75

### multi N=2 ablation hr_ir0_b75_dbg: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 75/75

### multi N=2 ablation ws_ir0_b1_dbg_i2: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 48/48

### multi N=2 ablation ws_ir0_b32_dbg_i2: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 58/58

### multi N=2 ablation ws_ir0_b75_dbg: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 75/75

### multi N=2 ablation ws_ir0_b8_dbg_i2: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 48/48

### multi N=2 ablation ws_ir100_b1_dbg_i2: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 48/48

### multi N=2 ablation ws_ir100_b32_dbg_i2: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3109 min 756.3109 max 756.3109; success 47/47

### multi N=2 ablation ws_ir100_b8_dbg_i2: IPOPT 756.3109; jaxipm median 756.3109 mean 756.3111 min 756.3109 max 756.3219; success 47/47

### multi N=4: one fixed problem solved 75× (jaxipm) / 75× (IPOPT)

- IPOPT objective 1537.7159 (std 2.27e-13); jaxipm median 1537.7159, mean 1531.6012, min 1080.0000, max 1537.7159; jaxipm within 1e-4 rel of IPOPT: 98.7%; jaxipm success flags 74/75

### multi N=4 ablation hr_ir0_b75_dbg: IPOPT 1537.7159; jaxipm median 1537.6883 mean 1537.7019 min 1537.6883 max 1537.7159; success 75/75

### multi N=4 ablation ws_ir0_b1_dbg_i2: IPOPT 1537.7159; jaxipm median 1537.7159 mean 1537.7027 min 1537.6883 max 1537.7159; success 48/48

### multi N=4 ablation ws_ir0_b32_dbg_i2: IPOPT 1537.7159; jaxipm median 1537.7159 mean 1537.7029 min 1537.6883 max 1537.7159; success 64/64

### multi N=4 ablation ws_ir0_b75_dbg: IPOPT 1537.7159; jaxipm median 1537.7159 mean 1537.7026 min 1537.6883 max 1537.7159; success 75/75

### multi N=4 ablation ws_ir0_b8_dbg_i2: IPOPT 1537.7159; jaxipm median 1537.7021 mean 1537.7021 min 1537.6883 max 1537.7159; success 48/48

### multi N=4 ablation ws_ir100_b1_dbg_i2: IPOPT 1537.7159; jaxipm median 1537.7159 mean 2028.7064 min 1537.6883 max 4320.0049; success 14/17

### multi N=4 ablation ws_ir100_b8_dbg_i2: IPOPT 1537.7159; jaxipm median 1537.7159 mean 1790.6921 min 1537.7159 max 4320.0550; success 10/11
