# Run K — fusion cost profile of one fused iteration

Device (GPU) time per fused iteration attributed to solver regions via `jax.named_scope` markers (jaxipm/utils/kscope.py) and the XLA profiler; cuDSS custom calls counted per iteration. Wall = median wall-clock of the jitted fused body (block_until_ready). Fusion tax = share of device time spent on branch-unique work (SFR trial, WD/SOC init, SOC loop, backtracking loop) that an oracle body told 'full step' would skip.

| region | multi_q2_b75 | nav_s90_b1 | nav_s90_b500 | track_v1.5_b250 |
|---|---|---|---|---|
| derivative evaluation (`pp_derivs`) | 0.896 ms (1.7%) | 0.318 ms (8.9%) | 1.197 ms (1.5%) | 0.811 ms (0.7%) |
| KKT assembly (`kkt_assembly`) | 0.505 ms (0.9%) | 0.108 ms (3.0%) | 1.918 ms (2.4%) | 0.750 ms (0.6%) |
| LS-multiplier factor+solve (`ls_mult_factor_solve`) | 0.003 ms (0.0%) | 0.002 ms (0.1%) | 0.005 ms (0.0%) | 0.003 ms (0.0%) |
| inertia-corr. loop (factorizations) (`ic_loop`) | 0.264 ms (0.5%) | 0.042 ms (1.2%) | 2.030 ms (2.6%) | 0.810 ms (0.7%) |
| quantities (`pp_quantities`) | 2.035 ms (3.8%) | 0.216 ms (6.0%) | 0.792 ms (1.0%) | 2.480 ms (2.1%) |
| mu update (`pp_mu_update`) | 2.174 ms (4.0%) | 0.474 ms (13.3%) | 2.405 ms (3.1%) | 2.995 ms (2.6%) |
| pre-mu misc (`pre_mu_misc`) | 0.105 ms (0.2%) | 0.021 ms (0.6%) | 0.272 ms (0.3%) | 0.248 ms (0.2%) |
| post-mu RHS (`post_mu_rhs`) | 0.012 ms (0.0%) | 0.007 ms (0.2%) | 0.031 ms (0.0%) | 0.021 ms (0.0%) |
| step transform (`post_mu_transform`) | 0.014 ms (0.0%) | 0.007 ms (0.2%) | 0.016 ms (0.0%) | 0.017 ms (0.0%) |
| first trial evaluation (`search_ls_first`) | 0.846 ms (1.6%) | 0.116 ms (3.2%) | 0.839 ms (1.1%) | 1.287 ms (1.1%) |
| SFR trial (branch-unique) (`search_sfr_setup`) | 5.934 ms (11.0%) | 0.418 ms (11.7%) | 1.146 ms (1.5%) | 7.559 ms (6.5%) |
| WD/SOC init (branch-unique) (`search_wd_socinit`) | 0.271 ms (0.5%) | 0.051 ms (1.4%) | 0.168 ms (0.2%) | 0.115 ms (0.1%) |
| SOC loop (branch-unique) (`search_soc`) | 21.155 ms (39.3%) | 0.044 ms (1.2%) | 14.817 ms (18.9%) | 66.151 ms (57.0%) |
| backtracking loop (branch-unique) (`search_bt`) | 3.813 ms (7.1%) | 0.001 ms (0.0%) | 1.457 ms (1.9%) | 5.203 ms (4.5%) |
| search select/state write (`search_select`) | 0.031 ms (0.1%) | 0.007 ms (0.2%) | 0.045 ms (0.1%) | 0.036 ms (0.0%) |
| post select/termination (`pp_select_term`) | 0.098 ms (0.2%) | 0.035 ms (1.0%) | 0.143 ms (0.2%) | 0.122 ms (0.1%) |
| resto/init bookkeeping (`pp_resto_init`) | 0.391 ms (0.7%) | 0.170 ms (4.8%) | 0.731 ms (0.9%) | 0.493 ms (0.4%) |
| scatter + hot-restart inject (`hr_restart`) | 0.007 ms (0.0%) | 0.007 ms (0.2%) | 0.053 ms (0.1%) | 0.026 ms (0.0%) |
| cuDSS factorization kernels (`cudss_factorize`) | 9.983 ms (18.5%) | 0.627 ms (17.5%) | 32.473 ms (41.4%) | 17.445 ms (15.0%) |
| cuDSS triangular-solve kernels (`cudss_solve`) | 1.750 ms (3.2%) | 0.334 ms (9.3%) | 8.098 ms (10.3%) | 3.925 ms (3.4%) |
| cuDSS symbolic-analysis kernels (`cudss_analysis`) | 1.991 ms (3.7%) | 0.060 ms (1.7%) | 5.141 ms (6.6%) | 2.830 ms (2.4%) |
| memcpy / memset (`memcpy`) | 1.217 ms (2.3%) | 0.425 ms (11.9%) | 3.627 ms (4.6%) | 2.052 ms (1.8%) |
| unscoped XLA kernels (`unscoped`) | 0.364 ms (0.7%) | 0.088 ms (2.5%) | 0.944 ms (1.2%) | 0.634 ms (0.5%) |
| **device total / iteration** | **53.86 ms** | **3.58 ms** | **78.35 ms** | **116.01 ms** |
| wall / iteration (median) | 55.79 ms | 11.97 ms | 87.83 ms | 132.23 ms |
| linear algebra (LS + IC loop + solves) | 13.99 ms (26.0%) | 1.06 ms (29.8%) | 47.75 ms (60.9%) | 25.01 ms (21.6%) |
| branch-unique work = fusion tax | 31.17 ms (57.9%) | 0.51 ms (14.4%) | 17.59 ms (22.4%) | 79.03 ms (68.1%) |
| kernels / iteration | 2015 | 1442 | 2404 | 3148 |

## cuDSS custom calls per fused iteration (count, device ms)

| target | multi_q2_b75 | nav_s90_b1 | nav_s90_b500 | track_v1.5_b250 |
|---|---|---|---|---|
| `cudss::adjncy_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.008 ms) | 0.10 × (0.004 ms) |
| `cudss::blocks_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.004 ms) | 0.10 × (0.002 ms) |
| `cudss::bwd_ker` | 13.80 × (0.927 ms) | 8.20 × (0.186 ms) | 18.00 × (3.882 ms) | 18.00 × (1.928 ms) |
| `cudss::compute_hybrid_minimum_chunk_size_ker` | 0.10 × (0.013 ms) | 0.00 × (0.000 ms) | 0.10 × (0.054 ms) | 0.10 × (0.028 ms) |
| `cudss::copy_and_convert_device_device_ker` | 6.60 × (0.010 ms) | 2.60 × (0.003 ms) | 12.40 × (0.045 ms) | 9.60 × (0.023 ms) |
| `cudss::copy_csr_columns_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `cudss::copy_matrix_ker` | 5.40 × (0.114 ms) | 3.30 × (0.004 ms) | 8.30 × (0.976 ms) | 6.90 × (0.282 ms) |
| `cudss::count_dep_fwd_bwd_ker` | 0.20 × (0.005 ms) | 0.00 × (0.000 ms) | 0.20 × (0.022 ms) | 0.20 × (0.011 ms) |
| `cudss::csc_rows_ker` | 1.00 × (0.152 ms) | 0.00 × (0.000 ms) | 1.00 × (0.909 ms) | 1.00 × (0.468 ms) |
| `cudss::define_superpanel_ker` | 0.10 × (0.067 ms) | 0.00 × (0.000 ms) | 0.10 × (0.013 ms) | 0.10 × (0.024 ms) |
| `cudss::dependency_map_ker` | 1.00 × (0.044 ms) | 0.00 × (0.000 ms) | 1.00 × (0.135 ms) | 1.00 × (0.073 ms) |
| `cudss::diag_ker` | 6.90 × (0.039 ms) | 5.10 × (0.013 ms) | 9.00 × (0.164 ms) | 9.00 × (0.084 ms) |
| `cudss::factorize_ker` | 5.40 × (7.469 ms) | 1.30 × (0.009 ms) | 8.30 × (31.678 ms) | 6.90 × (16.434 ms) |
| `cudss::factorize_v3_ker` | 5.40 × (2.514 ms) | 3.30 × (0.618 ms) | 8.30 × (0.796 ms) | 6.90 × (1.011 ms) |
| `cudss::fwd_bwd_order_step_1_ker` | 3.30 × (0.008 ms) | 0.00 × (0.000 ms) | 2.30 × (0.018 ms) | 2.40 × (0.011 ms) |
| `cudss::fwd_bwd_order_step_2_ker` | 5.40 × (0.017 ms) | 0.00 × (0.000 ms) | 4.20 × (0.049 ms) | 4.30 × (0.028 ms) |
| `cudss::fwd_ker` | 13.80 × (0.808 ms) | 8.20 × (0.144 ms) | 18.00 × (4.181 ms) | 18.00 × (1.974 ms) |
| `cudss::independent_ker` | 5.40 × (0.012 ms) | 3.30 × (0.019 ms) | 8.30 × (0.017 ms) | 6.90 × (0.014 ms) |
| `cudss::map_ker` | 0.10 × (0.051 ms) | 0.00 × (0.000 ms) | 0.10 × (0.205 ms) | 0.10 × (0.096 ms) |
| `cudss::map_offsets_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.005 ms) | 0.10 × (0.002 ms) |
| `cudss::modify_update_ker` | 0.10 × (0.016 ms) | 0.00 × (0.000 ms) | 0.10 × (0.069 ms) | 0.10 × (0.033 ms) |
| `cudss::nnz_count_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `cudss::nnz_per_col_ker` | 0.20 × (0.228 ms) | 0.00 × (0.000 ms) | 0.20 × (0.435 ms) | 0.20 × (0.255 ms) |
| `cudss::offsets_ker` | 0.10 × (0.001 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `cudss::perm_ker` | 13.80 × (0.034 ms) | 10.20 × (0.013 ms) | 18.00 × (0.129 ms) | 18.00 × (0.073 ms) |
| `cudss::plain_map_ker` | 5.40 × (0.012 ms) | 3.30 × (0.006 ms) | 8.30 × (0.018 ms) | 6.90 × (0.014 ms) |
| `cudss::radix_sort_ker` | 0.40 × (0.115 ms) | 0.00 × (0.000 ms) | 0.40 × (0.397 ms) | 0.40 × (0.204 ms) |
| `cudss::return_lu_diag_ker` | 3.30 × (0.015 ms) | 1.30 × (0.002 ms) | 6.20 × (0.339 ms) | 4.80 × (0.043 ms) |
| `cudss::set_default_ker` | 0.10 × (0.000 ms) | 0.00 × (0.000 ms) | 0.10 × (0.000 ms) | 0.10 × (0.000 ms) |
| `cudss::supernode_dependant_ker` | 0.10 × (0.000 ms) | 0.00 × (0.000 ms) | 0.10 × (0.000 ms) | 0.10 × (0.000 ms) |
| `cudss::supernode_map_ker` | 0.10 × (0.004 ms) | 0.00 × (0.000 ms) | 0.10 × (0.016 ms) | 0.10 × (0.008 ms) |
| `cudss::supernode_map_offsets_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.009 ms) | 0.10 × (0.004 ms) |
| `cudss::trans_columns_ker` | 0.10 × (0.007 ms) | 0.00 × (0.000 ms) | 0.10 × (0.030 ms) | 0.10 × (0.015 ms) |
| `cudss::trans_nnz_per_row_ker` | 0.10 × (0.007 ms) | 0.00 × (0.000 ms) | 0.10 × (0.030 ms) | 0.10 × (0.015 ms) |
| `cudss::upd_marker_bwd_ker` | 6.90 × (0.010 ms) | 3.10 × (0.004 ms) | 9.00 × (0.013 ms) | 9.00 × (0.012 ms) |
| `cudss::updates_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.004 ms) | 0.10 × (0.002 ms) |
| `cudss::updates_offsets_ker` | 0.10 × (0.002 ms) | 0.00 × (0.000 ms) | 0.10 × (0.003 ms) | 0.10 × (0.002 ms) |
| `cudss::xadj_ker` | 0.10 × (0.000 ms) | 0.00 × (0.000 ms) | 0.10 × (0.001 ms) | 0.10 × (0.001 ms) |
| `offsets_par_ker` | 0.60 × (1.013 ms) | 0.00 × (0.000 ms) | 0.60 × (1.055 ms) | 0.60 × (1.019 ms) |

Warm state: `solve_throughput` run for `warm_solves` solutions before profiling (members at mixed iteration counts, range multi_q2_b75: [0, 169], nav_s90_b1: [15, 15], nav_s90_b500: [0, 122], track_v1.5_b250: [0, 127]).
