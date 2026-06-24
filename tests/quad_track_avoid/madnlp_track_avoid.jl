
ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "1")

using ExaModels, MadNLP, MadNLPGPU, CUDA, NPZ, ProgressMeter, JSON

# Reference/initial-condition helpers are shared with the Python solvers (single
# source of truth in tests/quad_track_avoid/initialization.py) and called via
# PythonCall. Point PythonCall at an interpreter that has numpy; override
# JULIA_PYTHONCALL_EXE if your numpy-enabled python lives elsewhere.
get!(ENV, "JULIA_CONDAPKG_BACKEND", "Null")
get!(ENV, "JULIA_PYTHONCALL_EXE", "/opt/Miniconda/bin/python3")
using PythonCall

CUDA.allowscalar(false)
println("Using GPU: ", CUDA.device())

# ─────────────────────────────────────────────────────────────────────────────
# Shared quadcopter dynamics (continuous-time ẋ) + parameters from quad_params.json.
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "quad_examodels_dynamics.jl"))
const qp = (; (Symbol(k) => v for (k, v) in
               JSON.parsefile(joinpath(@__DIR__, "..", "quad_params.json")))...)

# Test-specific parameters (state bounds, obstacles, reference) from test_params.json.
# JSON has no infinity, so `null` denotes an unbounded (±Inf) entry — matching the
# Python ipopt/jaxipm convention in this directory.
const _tp = JSON.parsefile(joinpath(@__DIR__, "test_params.json"))
const x_lb = [v === nothing ? -Inf : Float64(v) for v in _tp["x_lb"]]
const x_ub = [v === nothing ?  Inf : Float64(v) for v in _tp["x_ub"]]

const nx = 13
const nu = 4
const N  = Int(_tp["N_horizon"])
const Ts = Float64(_tp["Ts"])

# Cost weights
const Q_cost = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Reference speeds to sweep (pringle amplitudes live with the shared Python helpers).
const AVG_VELS = [Float64(v) for v in _tp["avg_vel"]]

# Obstacle definitions from test_params.json: (xc0, yc0, r, ax, ay, fx, fy, px, py)
const obs_defs = [
    (o["xc0"], o["yc0"], o["r"], o["ax"], o["ay"], o["fx"], o["fy"], o["px"], o["py"])
    for o in _tp["obstacles"]
]

# ─────────────────────────────────────────────────────────────────────────────
# Reference trajectory + random initial conditions — shared Python helpers.
#
# The pringle reference and the obstacle-rejection sampler live in exactly ONE
# place (tests/quad_track_avoid/initialization.py) and are reused by all three
# solvers. We call them through PythonCall so madnlp solves the *identical*
# random instances as ipopt/jaxipm (same NumPy PCG64 stream for a given seed).
#
# NOTE: initialization.py loads test_params.json with a cwd-relative path, so
# (like the Python solver scripts) run this from the repo root.
# ─────────────────────────────────────────────────────────────────────────────
pyimport("sys").path.insert(0, abspath(joinpath(@__DIR__, "..", "..")))
const _pyinit = pyimport("tests.quad_track_avoid.initialization")

# Generate N_runs random valid (x0, xr, s) triples with obstacle rejection at
# t=0, via the shared Python sampler. Returns the same tuple as before.
function generate_random_inits(N_runs::Int, delta_std::Float64, avg_vel; seed = 0)
    res = _pyinit.generate_random_inits(N_runs, N, Ts, avg_vel;
                                        delta_std = delta_std, seed = seed)
    return (pyconvert(Array{Float64,2}, res[0]),   # all_x0  (N_runs, nx)
            pyconvert(Array{Float64,3}, res[1]),   # all_xr  (N_runs, N, nx)
            pyconvert(Array{Float64,1}, res[2]),   # all_s   (N_runs,)
            pyconvert(Int,              res[3]))   # n_rejects
end

# ─────────────────────────────────────────────────────────────────────────────
# Build ExaModel
# ─────────────────────────────────────────────────────────────────────────────
function build_model(x0_init, xr; backend = nothing)
    c = ExaCore(Float64; backend = backend)

    w_hover = 522.9847140714692
    n_x_vars = N * nx
    n_u_vars = (N - 1) * nu
    n_total  = n_x_vars + n_u_vars

    # Initial values and bounds
    x0_vals = zeros(n_total)
    lvar = fill(-Inf, n_total)
    uvar = fill( Inf, n_total)

    # Linear interpolation from start xyz to [0, 0, -2], q0=1, rest zero
    start_xyz = [x0_init[1], x0_init[2], x0_init[3]]
    end_xyz   = [0.0, 0.0, -2.0]
    for k in 1:N
        base = (k - 1) * nx
        t_frac = (k - 1) / (N - 1)
        for d in 1:3
            x0_vals[base + d] = (1 - t_frac) * start_xyz[d] + t_frac * end_xyz[d]
        end
        x0_vals[base + 4] = 1.0  # q0 = 1
        # entries 5..13 stay 0
        # State bounds
        for i in 1:nx
            lvar[base + i] = x_lb[i]
            uvar[base + i] = x_ub[i]
        end
    end
    for k in 1:(N-1)
        base = n_x_vars + (k - 1) * nu
        for j in 1:nu
            x0_vals[base + j] = w_hover
            lvar[base + j] = qp.minWmotor
            uvar[base + j] = qp.maxWmotor
        end
    end

    z = variable(c, n_total; start = x0_vals, lvar = lvar, uvar = uvar)

    xi(k, i) = z[(k-1)*nx + i]
    uj(k, j) = z[n_x_vars + (k-1)*nu + j]

    # --- Objective: tracking cost Q[i] * (X[i,k] - Xr[i,k])^2 ---
    obj_data = [(k, i, Q_cost[i], xr[k, i]) for k in 1:N for i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * (xi(k, i) - xr_val)^2 for (k, i, w, xr_val) in obj_data)

    # --- Initial condition ---
    ic_data = [(i, x0_init[i]) for i in 1:nx]
    constraint(c, xi(1, i) - x0v for (i, x0v) in ic_data; lcon = 0.0, ucon = 0.0)

    # --- Dynamics: explicit-Euler step on the shared continuous-time ẋ ---
    #     X[:,k+1] - X[:,k] - f(X[:,k], U[:,k])·Ts == 0   (one constraint per state)
    dyn_indices = [(k,) for k in 1:N-1]
    for i in 1:nx
        constraint(c, xi(k+1, i) - xi(k, i) - quad_xdot(xi, uj, k, qp)[i] * Ts
                   for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
    end

    # --- Quaternion norm band: 0.5 <= ||q||^2 <= 1.5 (two-sided, matches
    # jaxipm's loosened bounds).
    quat_data = [(k,) for k in 1:N]
    constraint(c,
        xi(k,4)^2 + xi(k,5)^2 + xi(k,6)^2 + xi(k,7)^2
        for (k,) in quat_data; lcon = 0.5, ucon = 1.5)

    # --- Time-varying obstacle avoidance: (x-xc(t))^2 + (y-yc(t))^2 >= r^2 ---
    obs_con_data = Tuple{Int,Float64,Float64,Float64}[]
    for k in 1:N
        tk = (k - 1) * Ts
        for (xc0, yc0, r, ax, ay, fx, fy, px, py) in obs_defs
            xc = xc0 + ax * sin(fx * tk + px)
            yc = yc0 + ay * sin(fy * tk + py)
            push!(obs_con_data, (k, xc, yc, r^2))
        end
    end
    constraint(c,
        (xi(k, 1) - xc)^2 + (xi(k, 2) - yc)^2 - r2
        for (k, xc, yc, r2) in obs_con_data; lcon = 0.0, ucon = Inf)

    return ExaModel(c)
end

# ─────────────────────────────────────────────────────────────────────────────
# Solve & save
# ─────────────────────────────────────────────────────────────────────────────
function solve_once(nlp; print_level=MadNLP.INFO)
    solver = MadNLPSolver(nlp;
        kkt_system               = MadNLP.SparseCondensedKKTSystem,
        linear_solver            = MadNLPGPU.CUDSSSolver,
        cudss_algorithm          = MadNLP.CHOLESKY,
        equality_treatment       = MadNLP.RelaxEquality,
        fixed_variable_treatment = MadNLP.RelaxBound,
        dual_initialized         = true,
        tol                      = 1e-6,
        mu_init                  = 1.0,
        first_hessian_perturbation = 1e-2,
        print_level              = print_level,
    )
    result = MadNLP.solve!(solver)
    return result
end

function run_avg_vel(average_vel, N_RUNS, delta_std)
    println("MadNLP: generating $N_RUNS random initial conditions (delta_std=$delta_std, avg_vel=$average_vel)")
    all_x0, all_xr, all_s, n_rejects = generate_random_inits(N_RUNS, delta_std, average_vel; seed = 0)
    println("MadNLP: rejected $n_rejects samples (kept $N_RUNS)")

    println("Building ExaModel on GPU (CUDABackend) — JIT warmup")
    nlp_warm = build_model(all_x0[1, :], all_xr[1, :, :]; backend = CUDABackend())
    println("\nProblem dimensions:")
    println("  Variables:   $(nlp_warm.meta.nvar)")
    println("  Constraints: $(nlp_warm.meta.ncon)")

    CUDA.synchronize()
    t_jit = @elapsed begin
        _ = solve_once(nlp_warm; print_level=MadNLP.ERROR)
        CUDA.synchronize()
    end
    println("Warmup solve (incl. JIT): $(t_jit * 1000) ms")

    X_all    = fill(NaN, N_RUNS, N, nx)
    times    = zeros(Float64, N_RUNS)
    iters    = zeros(Int32, N_RUNS)
    obj_vals = fill(NaN, N_RUNS)
    success  = zeros(Bool, N_RUNS)
    starts   = zeros(N_RUNS, nx)

    println("\nRunning $N_RUNS sequential cold solves with random (x0, xr)")
    t0 = time()
    prog = Progress(N_RUNS; desc="MadNLP", showspeed=true)
    for i in 1:N_RUNS
        x0_i = all_x0[i, :]
        xr_i = all_xr[i, :, :]
        starts[i, :] = x0_i

        nlp_i = build_model(x0_i, xr_i; backend = CUDABackend())
        CUDA.synchronize()
        t1 = time()
        result = solve_once(nlp_i; print_level=MadNLP.ERROR)
        CUDA.synchronize()
        t2 = time()

        times[i] = t2 - t1
        iters[i] = result.iter
        success[i] = (result.status == MadNLP.SOLVE_SUCCEEDED)
        try
            obj_vals[i] = result.objective
        catch
            try
                obj_vals[i] = result.f
            catch
                # Leave as NaN if neither field exists.
            end
        end

        z_sol = Array(result.solution)
        if any(isnan, z_sol)
        else
            X_flat = z_sol[1:N*nx]
            X_all[i, :, :] = reshape(X_flat, nx, N)' |> collect
        end

        sr = sum(success[1:i]) / i
        next!(prog; showvalues=[(:succ, round(sr, digits=3))])
    end
    finish!(prog)
    total_time = time() - t0

    logs_dir = joinpath(@__DIR__, "logs")
    mkpath(logs_dir)
    out_path = joinpath(logs_dir, "madnlp_v$(round(average_vel, digits=1))_results.npz")
    npzwrite(out_path,
             Dict("X_all" => X_all,
                  "times" => times,
                  "iters" => iters,
                  "obj_vals" => obj_vals,
                  "success" => success,
                  "starts" => starts,
                  "all_x0" => all_x0,
                  "all_xr" => all_xr,
                  "all_s" => all_s,
                  "total_time" => [total_time],
                  "N" => [N],
                  "Ts" => [Ts],
                  "N_RUNS" => [N_RUNS],
                  "avg_vel" => [average_vel]))

    n_succ = sum(success)
    mean_solve_ms = n_succ > 0 ? mean(times[success]) * 1000 : 0.0
    println("\nMadNLP: saved $out_path")
    println("MadNLP: total_time=$(round(total_time, digits=2))s, success=$n_succ/$N_RUNS, mean_solve=$(round(mean_solve_ms, digits=2))ms")
end

function main()
    N_RUNS = Int(_tp["N_RUNS_seq"])
    delta_std = parse(Float64, get(ENV, "DELTA_STD", "0.5"))
    for average_vel in AVG_VELS
        println("\n================ avg_vel = $average_vel ================")
        run_avg_vel(average_vel, N_RUNS, delta_std)
    end
end

using Statistics
main()
