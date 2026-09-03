# Primary-table quality columns: final cost (mean ± sd over the pool) and violations at the returned point

Violations evaluated with jaxipm's own c(z)/d(z)/bounds for every solver (same functions, same scaling). Columns: ‖c‖∞ = max |equality residual| (dynamics + initial condition), ineq = max(d_L−d, d−d_U, 0), bound = max(x_L−z, z−x_U, 0); each reported as median / mean / max over successful solves. Cost sd is the spread over the problem pool (not over repeats). IPOPT P=64 = run D (pinned processes, tol 1e-8). IPOPT default = one IPOPT/MUMPS process using all cores (tol 1e-8, same pool). MadNLP = run E pool run.

| scenario | solver | n | success | cost mean ± sd | ‖c‖∞ med/mean/max | ineq viol | bound viol | note |
|---|---|---|---|---|---|---|---|---|
| nav 90° | IPOPT default (MUMPS, tol 1e-8, pool) | 2000 | 100.0% | 516.953 ± 20.286 | 1.6e-12 / 3.9e-10 / 8.9e-09 | 9.9e-09 / 8.7e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 | one process, all cores |
| nav 90° | IPOPT P=64 (run D, tol 1e-8) | 2000 | 100.0% | 516.953 ± 20.286 | 1.6e-12 / 3.9e-10 / 8.9e-09 | 9.9e-09 / 8.7e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |  |
| nav 90° | MadNLP (tol 1e-8, pool) | 2000 | 100.0% | 516.994 ± 20.302 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 2.0e-08 / 1.8e-08 / 2.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| nav 90° | jaxipm-IL (ILB on) | 2000 | 100.0% | 517.161 ± 24.063 | – | – | – | paper npz (cost) |
| nav 90° | jaxipm-IL census run (HR dbg) | 2000 | 100.0% | 516.479 ± 20.225 | 3.5e-13 / 1.5e-11 / 7.7e-09 | 1.0e-08 / 9.0e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 | same config, DEBUG run; violations from z_all |
| nav 90° | jaxipm-WS (ILB off, dbg) | 2000 | 100.0% | 516.164 ± 20.159 | 3.4e-13 / 1.6e-11 / 9.7e-09 | 1.0e-08 / 8.8e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |  |
| nav 180° | IPOPT default (MUMPS, tol 1e-8, pool) | 2000 | 100.0% | 507.571 ± 17.143 | 9.0e-13 / 2.0e-10 / 9.9e-09 | 0.0e+00 / 4.4e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 | one process, all cores |
| nav 180° | IPOPT P=64 (run D, tol 1e-8) | 2000 | 100.0% | 507.571 ± 17.143 | 9.0e-13 / 2.0e-10 / 9.9e-09 | 0.0e+00 / 4.4e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |  |
| nav 180° | MadNLP (tol 1e-8, pool) | 2000 | 100.0% | 507.591 ± 17.162 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 0.0e+00 / 8.8e-09 / 2.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| nav 180° | jaxipm-IL (ILB on) | 2000 | 100.0% | 506.165 ± 20.097 | – | – | – | paper npz (cost) |
| nav 180° | jaxipm-IL census run (HR dbg) | 2000 | 100.0% | 506.779 ± 16.479 | 3.4e-13 / 1.5e-11 / 9.9e-09 | 0.0e+00 / 4.6e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 | same config, DEBUG run; violations from z_all |
| nav 180° | jaxipm-WS (ILB off, dbg) | 1999 | 100.0% | 506.816 ± 17.073 | 3.3e-13 / 1.8e-11 / 8.3e-09 | 0.0e+00 / 4.4e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |  |
| track v̄=1.0 | IPOPT default (MUMPS, tol 1e-8, pool) | 1000 | 100.0% | 265.709 ± 98.667 | 8.7e-11 / 7.8e-10 / 9.7e-09 | 9.9e-09 / 9.1e-09 / 1.0e-08 | 8.9e-06 / 8.0e-06 / 9.2e-06 | one process, all cores |
| track v̄=1.0 | IPOPT P=64 (run D, tol 1e-8) | 1000 | 100.0% | 265.709 ± 98.667 | 8.7e-11 / 7.8e-10 / 9.7e-09 | 9.9e-09 / 9.1e-09 / 1.0e-08 | 8.9e-06 / 8.0e-06 / 9.2e-06 |  |
| track v̄=1.0 | MadNLP (tol 1e-8, pool) | 1000 | 100.0% | 261.701 ± 92.033 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 2.0e-08 / 1.8e-08 / 2.0e-08 | 9.1e-06 / 8.3e-06 / 9.2e-06 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| track v̄=1.0 | jaxipm-IL (ILB on) | 1000 | 100.0% | 253.317 ± 92.583 | – | – | – | paper npz (cost) |
| track v̄=1.0 | jaxipm-IL census run (HR dbg) | 1000 | 100.0% | 245.593 ± 75.545 | 1.0e-13 / 1.2e-11 / 2.2e-09 | 1.0e-08 / 8.7e-09 / 1.0e-08 | 8.9e-09 / 7.9e-09 / 9.6e-09 | same config, DEBUG run; violations from z_all |
| track v̄=1.0 | jaxipm-WS (ILB off, dbg) | 993 | 100.0% | 249.396 ± 79.394 | 9.5e-14 / 1.8e-11 / 7.3e-09 | 1.0e-08 / 8.8e-09 / 1.0e-08 | 8.9e-09 / 7.9e-09 / 9.7e-09 |  |
| track v̄=1.5 | IPOPT default (MUMPS, tol 1e-8, pool) | 1000 | 100.0% | 556.694 ± 140.884 | 6.9e-11 / 9.0e-10 / 9.8e-09 | 9.9e-09 / 1.0e-08 / 1.5e-08 | 9.1e-06 / 8.9e-06 / 9.2e-06 | one process, all cores |
| track v̄=1.5 | IPOPT P=64 (run D, tol 1e-8) | 1000 | 100.0% | 556.694 ± 140.884 | 6.9e-11 / 9.0e-10 / 9.8e-09 | 9.9e-09 / 1.0e-08 / 1.5e-08 | 9.1e-06 / 8.9e-06 / 9.2e-06 |  |
| track v̄=1.5 | MadNLP (tol 1e-8, pool) | 1000 | 100.0% | 555.812 ± 138.326 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 2.0e-08 / 2.1e-08 / 3.0e-08 | 9.2e-06 / 9.1e-06 / 9.2e-06 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| track v̄=1.5 | jaxipm-IL (ILB on) | 1000 | 100.0% | 547.688 ± 136.377 | – | – | – | paper npz (cost) |
| track v̄=1.5 | jaxipm-IL census run (HR dbg) | 1000 | 100.0% | 545.524 ± 123.372 | 1.5e-13 / 2.5e-11 / 2.4e-09 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.6e-09 / 9.3e-09 / 9.9e-09 | same config, DEBUG run; violations from z_all |
| track v̄=1.5 | jaxipm-WS (ILB off, dbg) | 968 | 100.0% | 543.564 ± 123.421 | 1.4e-13 / 2.7e-11 / 3.4e-09 | 1.0e-08 / 9.9e-09 / 1.0e-08 | 9.5e-09 / 9.3e-09 / 9.9e-09 |  |
| track v̄=2.0 | IPOPT default (MUMPS, tol 1e-8, pool) | 1000 | 100.0% | 1146.663 ± 126.935 | 3.6e-12 / 2.1e-10 / 4.6e-09 | 1.5e-08 / 1.5e-08 / 1.5e-08 | 9.2e-06 / 9.2e-06 / 9.2e-06 | one process, all cores |
| track v̄=2.0 | IPOPT P=64 (run D, tol 1e-8) | 1000 | 100.0% | 1146.663 ± 126.935 | 3.6e-12 / 2.1e-10 / 4.6e-09 | 1.5e-08 / 1.5e-08 / 1.5e-08 | 9.2e-06 / 9.2e-06 / 9.2e-06 |  |
| track v̄=2.0 | MadNLP (tol 1e-8, pool) | 1000 | 100.0% | 1147.484 ± 121.172 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 3.0e-08 / 3.0e-08 / 3.0e-08 | 9.2e-06 / 9.2e-06 / 9.2e-06 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| track v̄=2.0 | jaxipm-IL (ILB on) | 1000 | 100.0% | 1155.070 ± 151.507 | – | – | – | paper npz (cost) |
| track v̄=2.0 | jaxipm-IL census run (HR dbg) | 1000 | 100.0% | 1146.953 ± 131.923 | 2.4e-13 / 2.7e-11 / 8.0e-09 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.8e-09 / 9.8e-09 / 1.0e-08 | same config, DEBUG run; violations from z_all |
| track v̄=2.0 | jaxipm-WS (ILB off, dbg) | 858 | 100.0% | 1142.525 ± 128.938 | 2.3e-13 / 1.9e-11 / 2.6e-09 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.8e-09 / 9.8e-09 / 1.0e-08 |  |
| multi N=2 | IPOPT default (MUMPS, tol 1e-8, pool) | 75 | 100.0% | 756.311 ± 0.000 | 3.1e-11 / 3.1e-11 / 3.1e-11 | 9.5e-09 / 9.5e-09 / 9.5e-09 | 9.1e-06 / 9.1e-06 / 9.1e-06 | one process, all cores |
| multi N=2 | IPOPT P=64 (run D, tol 1e-8) | 320 | 100.0% | 756.311 ± 0.000 | 3.1e-11 / 3.1e-11 / 3.1e-11 | 9.5e-09 / 9.5e-09 / 9.5e-09 | 9.1e-06 / 9.1e-06 / 9.1e-06 |  |
| multi N=2 | MadNLP (tol 1e-8, pool) | 75 | 100.0% | 756.311 ± 0.000 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 9.1e-06 / 9.1e-06 / 9.2e-06 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| multi N=2 | jaxipm-IL (ILB on) | 75 | 98.7% | 756.311 ± 0.000 | – | – | – | paper npz (cost) |
| multi N=2 | jaxipm-IL census run (HR dbg) | 75 | 100.0% | 756.311 ± 0.000 | 2.8e-13 / 6.8e-13 / 9.7e-12 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.4e-09 / 9.4e-09 / 9.4e-09 | same config, DEBUG run; violations from z_all |
| multi N=2 | jaxipm-WS (ILB off, dbg) | 75 | 100.0% | 756.311 ± 0.000 | 2.2e-13 / 6.7e-13 / 1.0e-11 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.4e-09 / 9.4e-09 / 9.4e-09 |  |
| multi N=4 | IPOPT default (MUMPS, tol 1e-8, pool) | 75 | 100.0% | 1537.688 ± 0.000 | 9.2e-12 / 9.2e-12 / 9.2e-12 | 9.8e-09 / 9.8e-09 / 9.8e-09 | 9.1e-06 / 9.1e-06 / 9.1e-06 | one process, all cores |
| multi N=4 | IPOPT P=64 (run D, tol 1e-8) | 320 | 100.0% | 1537.716 ± 0.000 | 7.3e-09 / 7.3e-09 / 7.3e-09 | 9.8e-09 / 9.8e-09 / 9.8e-09 | 9.1e-06 / 9.1e-06 / 9.1e-06 |  |
| multi N=4 | MadNLP (tol 1e-8, pool) | 75 | 100.0% | 1537.704 ± 0.014 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 9.1e-06 / 9.1e-06 / 9.2e-06 | ‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified |
| multi N=4 | jaxipm-IL (ILB on) | 75 | 98.7% | 1537.704 ± 0.014 | – | – | – | paper npz (cost) |
| multi N=4 | jaxipm-IL census run (HR dbg) | 75 | 100.0% | 1537.702 ± 0.014 | 3.1e-12 / 1.1e-11 / 1.1e-10 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.5e-09 / 9.5e-09 / 9.5e-09 | same config, DEBUG run; violations from z_all |
| multi N=4 | jaxipm-WS (ILB off, dbg) | 75 | 100.0% | 1537.703 ± 0.014 | 1.2e-12 / 7.7e-12 / 4.8e-11 | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.5e-09 / 9.5e-09 / 9.5e-09 |  |
