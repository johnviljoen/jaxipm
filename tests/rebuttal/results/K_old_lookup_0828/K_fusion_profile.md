# Run K — fusion cost profile of one fused iteration

Device (GPU) time per fused iteration attributed to solver regions via `jax.named_scope` markers (jaxipm/utils/kscope.py) and the XLA profiler; cuDSS custom calls counted per iteration. Wall = median wall-clock of the jitted fused body (block_until_ready). Fusion tax = share of device time spent on branch-unique work (SFR trial, WD/SOC init, SOC loop, backtracking loop) that an oracle body told 'full step' would skip.

| region | multi_q2_b75 | nav_s90_b1 | nav_s90_b500 | track_v1.5_b250 |
|---|---|---|---|---|
| derivative evaluation (`pp_derivs`) | 1.236 ms (2.1%) | 0.544 ms (15.2%) | 3.534 ms (4.3%) | 2.780 ms (2.5%) |
| KKT assembly (`kkt_assembly`) | 0.021 ms (0.0%) | – | 0.057 ms (0.1%) | 0.006 ms (0.0%) |
| LS-multiplier factor+solve (`ls_mult_factor_solve`) | 0.034 ms (0.1%) | 0.016 ms (0.4%) | 0.002 ms (0.0%) | 0.002 ms (0.0%) |
| inertia-corr. loop (factorizations) (`ic_loop`) | 1.295 ms (2.2%) | 0.022 ms (0.6%) | 1.843 ms (2.3%) | 2.875 ms (2.6%) |
| quantities (`pp_quantities`) | 0.028 ms (0.0%) | 0.008 ms (0.2%) | 0.054 ms (0.1%) | 0.227 ms (0.2%) |
| mu update (`pp_mu_update`) | 2.057 ms (3.5%) | 0.579 ms (16.2%) | 2.909 ms (3.6%) | 2.973 ms (2.7%) |
| pre-mu misc (`pre_mu_misc`) | 0.030 ms (0.1%) | 0.006 ms (0.2%) | 0.135 ms (0.2%) | 0.009 ms (0.0%) |
| post-mu RHS (`post_mu_rhs`) | 0.003 ms (0.0%) | – | – | 0.008 ms (0.0%) |
| first trial evaluation (`search_ls_first`) | 0.039 ms (0.1%) | 0.027 ms (0.8%) | 0.761 ms (0.9%) | 0.077 ms (0.1%) |
| SFR trial (branch-unique) (`search_sfr_setup`) | 0.055 ms (0.1%) | 0.043 ms (1.2%) | 0.076 ms (0.1%) | 0.105 ms (0.1%) |
| WD/SOC init (branch-unique) (`search_wd_socinit`) | – | – | – | 0.001 ms (0.0%) |
| SOC loop (branch-unique) (`search_soc`) | 0.256 ms (0.4%) | 0.144 ms (4.0%) | 0.697 ms (0.9%) | 0.426 ms (0.4%) |
| backtracking loop (branch-unique) (`search_bt`) | 7.223 ms (12.3%) | 0.594 ms (16.6%) | 19.725 ms (24.1%) | 5.230 ms (4.7%) |
| search select/state write (`search_select`) | 0.007 ms (0.0%) | – | – | 0.005 ms (0.0%) |
| post select/termination (`pp_select_term`) | 0.050 ms (0.1%) | 0.018 ms (0.5%) | 0.045 ms (0.1%) | 0.046 ms (0.0%) |
| resto/init bookkeeping (`pp_resto_init`) | 0.087 ms (0.1%) | 0.032 ms (0.9%) | 0.174 ms (0.2%) | 0.137 ms (0.1%) |
| scatter + hot-restart inject (`hr_restart`) | – | – | – | 0.001 ms (0.0%) |
| cuDSS factorization kernels (`cudss_factorize`) | 10.187 ms (17.4%) | 0.628 ms (17.6%) | 28.542 ms (34.9%) | 16.928 ms (15.1%) |
| cuDSS triangular-solve kernels (`cudss_solve`) | 1.805 ms (3.1%) | 0.334 ms (9.3%) | 8.097 ms (9.9%) | 3.919 ms (3.5%) |
| cuDSS symbolic-analysis kernels (`cudss_analysis`) | 2.001 ms (3.4%) | 0.060 ms (1.7%) | 4.945 ms (6.1%) | 2.826 ms (2.5%) |
| memcpy / memset (`memcpy`) | 1.220 ms (2.1%) | 0.422 ms (11.8%) | 3.128 ms (3.8%) | 1.980 ms (1.8%) |
| unscoped XLA kernels (`unscoped`) | 31.047 ms (52.9%) | 0.098 ms (2.7%) | 7.005 ms (8.6%) | 71.304 ms (63.7%) |
| **device total / iteration** | **58.68 ms** | **3.57 ms** | **81.73 ms** | **111.87 ms** |
| wall / iteration (median) | 55.57 ms | 11.03 ms | 82.58 ms | 127.89 ms |
| linear algebra (LS + IC loop + solves) | 15.32 ms (26.1%) | 1.06 ms (29.6%) | 43.43 ms (53.1%) | 26.55 ms (23.7%) |
| branch-unique work = fusion tax | 7.53 ms (12.8%) | 0.78 ms (21.9%) | 20.50 ms (25.1%) | 5.76 ms (5.2%) |
| kernels / iteration | 2101 | 1442 | 2592 | 3078 |

## cuDSS custom calls per fused iteration (count, device ms)

| target | multi_q2_b75 | nav_s90_b1 | nav_s90_b500 | track_v1.5_b250 |
|---|---|---|---|---|
| `cudss::adjncy_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.007 ms) | 0.10 × (0.005 ms) |
| `cudss::blocks_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.004 ms) | 0.10 × (0.002 ms) |
| `cudss::bwd_ker` | 14.20 × (0.956 ms) | 8.20 × (0.186 ms) | 18.00 × (3.882 ms) | 18.00 × (1.927 ms) |
| `cudss::compute_hybrid_minimum_chunk_size_ker` | 0.10 × (0.013 ms) | 0.00 × (0.000 ms) | 0.10 × (0.055 ms) | 0.10 × (0.028 ms) |
| `cudss::copy_and_convert_device_device_ker` | 6.80 × (0.010 ms) | 2.60 × (0.003 ms) | 10.40 × (0.038 ms) | 9.20 × (0.022 ms) |
| `cudss::copy_csr_columns_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `cudss::copy_matrix_ker` | 5.50 × (0.116 ms) | 3.30 × (0.004 ms) | 7.30 × (0.849 ms) | 6.70 × (0.284 ms) |
| `cudss::count_dep_fwd_bwd_ker` | 0.20 × (0.005 ms) | 0.00 × (0.000 ms) | 0.20 × (0.022 ms) | 0.20 × (0.011 ms) |
| `cudss::csc_rows_ker` | 1.00 × (0.152 ms) | 0.00 × (0.000 ms) | 1.00 × (0.909 ms) | 1.00 × (0.471 ms) |
| `cudss::define_superpanel_ker` | 0.10 × (0.067 ms) | 0.00 × (0.000 ms) | 0.10 × (0.013 ms) | 0.10 × (0.024 ms) |
| `cudss::dependency_map_ker` | 1.00 × (0.045 ms) | 0.00 × (0.000 ms) | 1.00 × (0.134 ms) | 1.00 × (0.073 ms) |
| `cudss::diag_ker` | 7.10 × (0.041 ms) | 5.10 × (0.013 ms) | 9.00 × (0.165 ms) | 9.00 × (0.082 ms) |
| `cudss::factorize_ker` | 5.50 × (7.617 ms) | 1.30 × (0.009 ms) | 7.30 × (27.837 ms) | 6.70 × (15.947 ms) |
| `cudss::factorize_v3_ker` | 5.50 × (2.570 ms) | 3.30 × (0.619 ms) | 7.30 × (0.705 ms) | 6.70 × (0.981 ms) |
| `cudss::fwd_bwd_order_step_1_ker` | 3.30 × (0.008 ms) | 0.00 × (0.000 ms) | 2.30 × (0.018 ms) | 2.40 × (0.011 ms) |
| `cudss::fwd_bwd_order_step_2_ker` | 5.40 × (0.017 ms) | 0.00 × (0.000 ms) | 4.20 × (0.049 ms) | 4.30 × (0.028 ms) |
| `cudss::fwd_ker` | 14.20 × (0.834 ms) | 8.20 × (0.144 ms) | 18.00 × (4.181 ms) | 18.00 × (1.968 ms) |
| `cudss::independent_ker` | 5.50 × (0.012 ms) | 3.30 × (0.019 ms) | 7.30 × (0.015 ms) | 6.70 × (0.013 ms) |
| `cudss::map_ker` | 0.10 × (0.051 ms) | 0.00 × (0.000 ms) | 0.10 × (0.205 ms) | 0.10 × (0.096 ms) |
| `cudss::map_offsets_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.005 ms) | 0.10 × (0.002 ms) |
| `cudss::modify_update_ker` | 0.10 × (0.016 ms) | 0.00 × (0.000 ms) | 0.10 × (0.069 ms) | 0.10 × (0.033 ms) |
| `cudss::nnz_count_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `cudss::nnz_per_col_ker` | 0.20 × (0.229 ms) | 0.00 × (0.000 ms) | 0.20 × (0.434 ms) | 0.20 × (0.255 ms) |
| `cudss::offsets_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `cudss::perm_ker` | 14.20 × (0.035 ms) | 10.20 × (0.013 ms) | 18.00 × (0.129 ms) | 18.00 × (0.069 ms) |
| `cudss::plain_map_ker` | 5.50 × (0.012 ms) | 3.30 × (0.006 ms) | 7.30 × (0.016 ms) | 6.70 × (0.013 ms) |
| `cudss::radix_sort_ker` | 0.40 × (0.116 ms) | 0.00 × (0.000 ms) | 0.40 × (0.397 ms) | 0.40 × (0.204 ms) |
| `cudss::return_lu_diag_ker` | 3.40 × (0.015 ms) | 1.30 × (0.002 ms) | 5.20 × (0.282 ms) | 4.60 × (0.041 ms) |
| `cudss::set_default_ker` | 0.10 × (0.000 ms) | 0.00 × (0.000 ms) | 0.10 × (0.000 ms) | 0.10 × (0.000 ms) |
| `cudss::supernode_dependant_ker` | 0.10 × (0.000 ms) | 0.00 × (0.000 ms) | 0.10 × (0.000 ms) | 0.10 × (0.000 ms) |
| `cudss::supernode_map_ker` | 0.10 × (0.004 ms) | 0.00 × (0.000 ms) | 0.10 × (0.016 ms) | 0.10 × (0.008 ms) |
| `cudss::supernode_map_offsets_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.009 ms) | 0.10 × (0.004 ms) |
| `cudss::trans_columns_ker` | 0.10 × (0.007 ms) | 0.00 × (0.000 ms) | 0.10 × (0.030 ms) | 0.10 × (0.015 ms) |
| `cudss::trans_nnz_per_row_ker` | 0.10 × (0.007 ms) | 0.00 × (0.000 ms) | 0.10 × (0.030 ms) | 0.10 × (0.015 ms) |
| `cudss::upd_marker_bwd_ker` | 7.10 × (0.011 ms) | 3.10 × (0.004 ms) | 9.00 × (0.013 ms) | 9.00 × (0.013 ms) |
| `cudss::updates_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.004 ms) | 0.10 × (0.002 ms) |
| `cudss::updates_offsets_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.003 ms) | 0.10 × (0.002 ms) |
| `cudss::xadj_ker` | 0.10 × (0.000 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `offsets_par_ker` | 0.60 × (1.013 ms) | 0.00 × (0.000 ms) | 0.60 × (1.055 ms) | 0.60 × (1.019 ms) |

Warm state: `solve_throughput` run for `warm_solves` solutions before profiling (members at mixed iteration counts, range multi_q2_b75: [4, 158], nav_s90_b1: [13, 13], nav_s90_b500: [0, 121], track_v1.5_b250: [0, 120]).
