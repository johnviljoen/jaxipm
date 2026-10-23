# The three paper quadrotor NLPs (nav, track-avoid, multi-swap) as CPU ExaModels.
#
# These are verbatim copies of the `build_model` functions in
#   tests/quad_nav_circle/madnlp_quad.jl
#   tests/quad_track_avoid/madnlp_track_avoid.jl
#   tests/quad_multi_swap/madnlp_multi_swap.jl
# (same variable layout, same cost, same constraints, same cold start), with the
# MadNLP-specific driver code stripped so the models can be handed to a different
# solver. Shared dynamics come from tests/quad_examodels_dynamics.jl, parameters
# from the per-task test_params.json, and the track-avoid instances from the
# shared Python sampler (tests/quad_track_avoid/initialization.py) via PythonCall,
# exactly as the MadNLP driver does.
#
# Include from any script; run from the repo root (initialization.py loads its
# JSON by a cwd-relative path).

using ExaModels, JSON

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO_ROOT, "tests", "quad_examodels_dynamics.jl"))
const qp = (; (Symbol(k) => v for (k, v) in
               JSON.parsefile(joinpath(REPO_ROOT, "tests", "quad_params.json")))...)

const nx = 13
const nu = 4
const w_hover = 522.9847140714692

_bound(v) = v === nothing ? Inf : Float64(v)

# ─────────────────────────────────────────────────────────────────────────────
# nav (tests/quad_nav_circle)
# ─────────────────────────────────────────────────────────────────────────────
const NAV = let tp = JSON.parsefile(joinpath(REPO_ROOT, "tests", "quad_nav_circle", "test_params.json"))
    (; N = Int(tp["N_horizon"]), Ts = Float64(tp["Ts"]),
       x_lb = [v === nothing ? -Inf : Float64(v) for v in tp["x_lb"]],
       x_ub = [_bound(v) for v in tp["x_ub"]],
       obs_xc = [Float64(o["xc"]) for o in tp["obstacles"]],
       obs_yc = [Float64(o["yc"]) for o in tp["obstacles"]],
       obs_r  = [Float64(o["r"]) + Float64(tp["quad_radius"]) for o in tp["obstacles"]],
       Q = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
       N_RUNS_seq = Int(tp["N_RUNS_seq"]), init_angles = [Float64(a) for a in tp["init_angles"]],
       radius = sqrt(4.0^2 + 4.0^2))
end

nav_x0_from_angle(theta::Float64) = [NAV.radius * cos(theta), NAV.radius * sin(theta), 0.0,
                                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

function nav_angles(sector_deg::Float64, n_runs::Int)
    sector_half = deg2rad(sector_deg) / 2.0
    sector_center = pi / 4.0
    return collect(range(sector_center - sector_half, stop = sector_center + sector_half, length = n_runs))
end

function build_nav(x0_ic_vec::Vector{Float64}; backend = nothing)
    N, Ts, Q_cost = NAV.N, NAV.Ts, NAV.Q
    c = ExaCore(Float64; backend = backend)
    start_xyz = x0_ic_vec[1:3]; end_xyz = [0.0, 0.0, -2.0]
    n_x_vars = N * nx; n_u_vars = (N - 1) * nu; n_total = n_x_vars + n_u_vars
    x0_vals = zeros(n_total); lvar = fill(-Inf, n_total); uvar = fill(Inf, n_total)
    for k in 1:N
        t = (k - 1) / (N - 1); base = (k - 1) * nx
        for d in 1:3
            x0_vals[base + d] = (1 - t) * start_xyz[d] + t * end_xyz[d]
        end
        x0_vals[base + 4] = 1.0
        for i in 1:nx
            lvar[base + i] = NAV.x_lb[i]; uvar[base + i] = NAV.x_ub[i]
        end
    end
    for k in 1:(N-1)
        base = n_x_vars + (k - 1) * nu
        for j in 1:nu
            x0_vals[base + j] = w_hover; lvar[base + j] = qp.minWmotor; uvar[base + j] = qp.maxWmotor
        end
    end
    z = variable(c, n_total; start = x0_vals, lvar = lvar, uvar = uvar)
    xi(k, i) = z[(k-1)*nx + i]
    uj(k, j) = z[n_x_vars + (k-1)*nu + j]
    obj_data = [(k, i, Q_cost[i]) for k in 1:N for i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * xi(k, i)^2 for (k, i, w) in obj_data)
    ic_data = [(i, x0_ic_vec[i]) for i in 1:nx]
    constraint(c, xi(1, i) - x0v for (i, x0v) in ic_data; lcon = 0.0, ucon = 0.0)
    dyn_indices = [(k,) for k in 1:N-1]
    for i in 1:nx
        constraint(c, xi(k+1, i) - xi(k, i) - quad_xdot(xi, uj, k, qp)[i] * Ts
                   for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
    end
    obs_data = [(k, NAV.obs_xc[j], NAV.obs_yc[j], NAV.obs_r[j]^2 * (1.0 + (k-1)*Ts*0.1))
                for k in 1:N-1 for j in 1:length(NAV.obs_xc)]
    constraint(c, (xi(k, 1) - xc)^2 + (xi(k, 2) - yc)^2 - r2m
               for (k, xc, yc, r2m) in obs_data; lcon = 0.0, ucon = Inf)
    return ExaModel(c)
end

# ─────────────────────────────────────────────────────────────────────────────
# track-avoid (tests/quad_track_avoid)
# ─────────────────────────────────────────────────────────────────────────────
const TRACK = let tp = JSON.parsefile(joinpath(REPO_ROOT, "tests", "quad_track_avoid", "test_params.json"))
    (; N = Int(tp["N_horizon"]), Ts = Float64(tp["Ts"]),
       x_lb = [v === nothing ? -Inf : Float64(v) for v in tp["x_lb"]],
       x_ub = [_bound(v) for v in tp["x_ub"]],
       Q = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
       avg_vels = [Float64(v) for v in tp["avg_vel"]],
       obs_defs = [(o["xc0"], o["yc0"], o["r"], o["ax"], o["ay"], o["fx"], o["fy"], o["px"], o["py"]) for o in tp["obstacles"]],
       N_RUNS_seq = Int(tp["N_RUNS_seq"]))
end

# Random instances via the shared Python sampler (identical PCG64 stream as ipopt/jaxipm/madnlp).
function track_random_inits(N_runs::Int, avg_vel::Float64; delta_std = 0.5, seed = 0)
    pyinit = Main.pyimport("tests.quad_track_avoid.initialization")
    res = pyinit.generate_random_inits(N_runs, TRACK.N, TRACK.Ts, avg_vel; delta_std = delta_std, seed = seed)
    return (Main.pyconvert(Array{Float64,2}, res[0]), Main.pyconvert(Array{Float64,3}, res[1]),
            Main.pyconvert(Array{Float64,1}, res[2]), Main.pyconvert(Int, res[3]))
end

function build_track(x0_init, xr; backend = nothing)
    N, Ts, Q_cost = TRACK.N, TRACK.Ts, TRACK.Q
    c = ExaCore(Float64; backend = backend)
    n_x_vars = N * nx; n_u_vars = (N - 1) * nu; n_total = n_x_vars + n_u_vars
    x0_vals = zeros(n_total); lvar = fill(-Inf, n_total); uvar = fill(Inf, n_total)
    start_xyz = x0_init[1:3]; end_xyz = [0.0, 0.0, -2.0]
    for k in 1:N
        base = (k - 1) * nx; t_frac = (k - 1) / (N - 1)
        for d in 1:3
            x0_vals[base + d] = (1 - t_frac) * start_xyz[d] + t_frac * end_xyz[d]
        end
        x0_vals[base + 4] = 1.0
        for i in 1:nx
            lvar[base + i] = TRACK.x_lb[i]; uvar[base + i] = TRACK.x_ub[i]
        end
    end
    for k in 1:(N-1)
        base = n_x_vars + (k - 1) * nu
        for j in 1:nu
            x0_vals[base + j] = w_hover; lvar[base + j] = qp.minWmotor; uvar[base + j] = qp.maxWmotor
        end
    end
    z = variable(c, n_total; start = x0_vals, lvar = lvar, uvar = uvar)
    xi(k, i) = z[(k-1)*nx + i]
    uj(k, j) = z[n_x_vars + (k-1)*nu + j]
    obj_data = [(k, i, Q_cost[i], xr[k, i]) for k in 1:N for i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * (xi(k, i) - xr_val)^2 for (k, i, w, xr_val) in obj_data)
    ic_data = [(i, x0_init[i]) for i in 1:nx]
    constraint(c, xi(1, i) - x0v for (i, x0v) in ic_data; lcon = 0.0, ucon = 0.0)
    dyn_indices = [(k,) for k in 1:N-1]
    for i in 1:nx
        constraint(c, xi(k+1, i) - xi(k, i) - quad_xdot(xi, uj, k, qp)[i] * Ts
                   for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
    end
    quat_data = [(k,) for k in 1:N]
    constraint(c, xi(k,4)^2 + xi(k,5)^2 + xi(k,6)^2 + xi(k,7)^2 for (k,) in quat_data; lcon = 0.5, ucon = 1.5)
    obs_con_data = Tuple{Int,Float64,Float64,Float64}[]
    for k in 1:N
        tk = (k - 1) * Ts
        for (xc0, yc0, r, ax, ay, fx, fy, px, py) in TRACK.obs_defs
            push!(obs_con_data, (k, xc0 + ax * sin(fx * tk + px), yc0 + ay * sin(fy * tk + py), r^2))
        end
    end
    constraint(c, (xi(k, 1) - xc)^2 + (xi(k, 2) - yc)^2 - r2
               for (k, xc, yc, r2) in obs_con_data; lcon = 0.0, ucon = Inf)
    return ExaModel(c)
end

# ─────────────────────────────────────────────────────────────────────────────
# multi-swap (tests/quad_multi_swap)
# ─────────────────────────────────────────────────────────────────────────────
const MULTI = let tp = JSON.parsefile(joinpath(REPO_ROOT, "tests", "quad_multi_swap", "test_params.json"))
    quad_radius = Float64(tp["quad_radius"]); safety = Float64(tp["safety"])
    min_dist = 2.0 * (2 * quad_radius + safety)
    (; N = Int(tp["N_horizon"]), Ts = Float64(tp["Ts"]),
       XY_MARGIN = Float64(tp["xy_margin"]), Z_BOUND = Float64(tp["z_bound"]),
       VEL_RATE_BOUND = Float64(tp["vel_rate_bound"]), MIN_DIST = min_dist, MIN_DIST_SQ = min_dist^2,
       Q = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
       R = Float64(tp["R"]), N_quads = [Int(q) for q in tp["N_quads"]],
       N_RUNS_seq = Int(tp["N_RUNS_seq"]), n_pool = Int(tp["N_RUNS_jaxipm"]))
end

const POOL_ALPHA = 0.6180339887498949
const POOL_BETA  = 0.41421356237309515
const POOL_HALF_RANGE = deg2rad(5.0)
pool_phi(k::Int) = POOL_HALF_RANGE * (2.0 * mod(k * POOL_ALPHA, 1.0) - 1.0)
pool_psi(k::Int) = pool_phi(k) + POOL_HALF_RANGE * (2.0 * mod(k * POOL_BETA, 1.0) - 1.0)

function quad_start(q::Int, Nq::Int, R::Float64, phi::Float64 = 0.0)
    θ = 2π * (q - 1) / Nq + phi
    return [R*cos(θ), R*sin(θ), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end
function quad_goal(q::Int, Nq::Int, R::Float64, psi::Float64 = 0.0)
    θ = 2π * (q - 1) / Nq + π + psi
    return [R*cos(θ), R*sin(θ), 0.0]
end

function build_multi(Nq::Int, R::Float64; phi::Float64 = 0.0, psi::Float64 = 0.0, backend = nothing)
    N, Ts, Q_cost = MULTI.N, MULTI.Ts, MULTI.Q
    c = ExaCore(Float64; backend = backend)
    stride_x = N * nx; stride_u = (N - 1) * nu
    n_x_vars = Nq * stride_x; n_u_vars = Nq * stride_u; n_total = n_x_vars + n_u_vars
    vrb = MULTI.VEL_RATE_BOUND
    x_lb = [-(R+MULTI.XY_MARGIN), -(R+MULTI.XY_MARGIN), -MULTI.Z_BOUND, -Inf, -Inf, -Inf, -Inf, -vrb, -vrb, -vrb, -vrb, -vrb, -vrb]
    x_ub = [ (R+MULTI.XY_MARGIN),  (R+MULTI.XY_MARGIN),  MULTI.Z_BOUND,  Inf,  Inf,  Inf,  Inf,  vrb,  vrb,  vrb,  vrb,  vrb,  vrb]
    starts = [quad_start(q, Nq, R, phi) for q in 1:Nq]
    goals  = [quad_goal(q, Nq, R, psi)  for q in 1:Nq]
    x0_vals = zeros(n_total); lvar = fill(-Inf, n_total); uvar = fill(Inf, n_total)
    for q in 1:Nq, k in 1:N
        base_k = (q - 1) * stride_x + (k - 1) * nx
        for i in 1:nx
            x0_vals[base_k + i] = starts[q][i]; lvar[base_k + i] = x_lb[i]; uvar[base_k + i] = x_ub[i]
        end
    end
    for q in 1:Nq, k in 1:(N-1)
        base_k = n_x_vars + (q - 1) * stride_u + (k - 1) * nu
        for j in 1:nu
            x0_vals[base_k + j] = w_hover; lvar[base_k + j] = qp.minWmotor; uvar[base_k + j] = qp.maxWmotor
        end
    end
    z = variable(c, n_total; start = x0_vals, lvar = lvar, uvar = uvar)
    xi(q, k, i) = z[(q-1)*stride_x + (k-1)*nx + i]
    uj(q, k, j) = z[n_x_vars + (q-1)*stride_u + (k-1)*nu + j]
    goal_val(q, i) = (i == 1 || i == 2 || i == 3) ? goals[q][i] : 0.0
    obj_data = [(q, k, i, Q_cost[i], goal_val(q, i)) for q in 1:Nq, k in 1:N, i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * (xi(q, k, i) - gval)^2 for (q, k, i, w, gval) in obj_data)
    ic_data = [(q, i, starts[q][i]) for q in 1:Nq, i in 1:nx]
    constraint(c, xi(q, 1, i) - x0v for (q, i, x0v) in ic_data; lcon = 0.0, ucon = 0.0)
    dyn_indices = [(k,) for k in 1:N-1]
    for q in 1:Nq
        xi_q = (k, i) -> xi(q, k, i)
        uj_q = (k, j) -> uj(q, k, j)
        for i in 1:nx
            constraint(c, xi_q(k+1, i) - xi_q(k, i) - quad_xdot(xi_q, uj_q, k, qp)[i] * Ts
                       for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
        end
    end
    if Nq >= 2
        pair_data = [(i, j, k) for i in 1:Nq for j in (i+1):Nq for k in 1:(N-1)]
        constraint(c, (xi(i, k, 1) - xi(j, k, 1))^2 + (xi(i, k, 2) - xi(j, k, 2))^2 + (xi(i, k, 3) - xi(j, k, 3))^2 - MULTI.MIN_DIST_SQ
                   for (i, j, k) in pair_data; lcon = 0.0, ucon = Inf)
    end
    return ExaModel(c), starts, goals
end

# multi-swap success check (same semantics as the MadNLP/CasADi drivers)
function multi_check_success(X_quad::Array{Float64,3}, goals; goal_tol = 0.3, collision_tol = 0.8 * MULTI.MIN_DIST)
    Nq = size(X_quad, 1)
    goal_ok = maximum(sqrt(sum((X_quad[q, end, 1:3] .- goals[q]).^2)) for q in 1:Nq) < goal_tol
    min_d = Inf
    for i in 1:Nq, j in (i+1):Nq, k in 1:size(X_quad, 2)
        min_d = min(min_d, sqrt(sum((X_quad[i, k, 1:3] .- X_quad[j, k, 1:3]).^2)))
    end
    return goal_ok && min_d > collision_tol
end
