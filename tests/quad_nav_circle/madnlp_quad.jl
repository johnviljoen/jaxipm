
ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "1")

using ExaModels, MadNLP, MadNLPGPU, CUDA, NPZ, ProgressMeter, Statistics, JSON

CUDA.allowscalar(false)

# ── Rebuttal hooks (run E). Defaults reproduce the paper driver exactly. ──────
#   MADNLP_TOL=1e-8        solver tolerance override
#   MADNLP_N_RUNS=2000     pool size override (use N_RUNS_jaxipm for the shared pool)
#   MADNLP_REPEATS=5       timed repeats of the whole pool after the warmup solve
#   MADNLP_OUT_SUFFIX=_x   appended to the results filename (never clobber paper npz)
const MADNLP_TOL     = parse(Float64, get(ENV, "MADNLP_TOL", "1e-8"))
const MADNLP_REPEATS = parse(Int, get(ENV, "MADNLP_REPEATS", "1"))
const MADNLP_SUFFIX  = get(ENV, "MADNLP_OUT_SUFFIX", "")
#   MADNLP_NO_BOUNDS=1     drop state + motor bounds (parity with jaxipm_quad_nav, which has +-Inf bounds)
const MADNLP_NO_BOUNDS = get(ENV, "MADNLP_NO_BOUNDS", "0") == "1"
MADNLP_NO_BOUNDS && println("MADNLP_NO_BOUNDS=1: state and motor bounds DISABLED")
function _madnlp_meta_write(out_path::String, extra::Dict)
    git(args...) = try strip(read(Cmd(["git", "-C", @__DIR__, args...]), String)) catch; "unknown" end
    d = Dict{String,Any}("tol" => MADNLP_TOL, "repeats" => MADNLP_REPEATS, "no_bounds" => MADNLP_NO_BOUNDS,
        "jaxipm_release_commit" => git("rev-parse", "HEAD"),
        "dirty" => git("status", "--porcelain") != "",
        "hostname" => gethostname(), "gpu" => string(CUDA.device()),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "julia" => string(VERSION), "argv" => ARGS, "script" => @__FILE__)
    merge!(d, extra)
    d["madnlp_status_legend"] = Dict(string(Int(st)) => string(st) for st in instances(MadNLP.Status))
    open(out_path * ".meta.json", "w") do io; JSON.print(io, d, 2); end
end
println("Using GPU: ", CUDA.device())

# ─────────────────────────────────────────────────────────────────────────────
# Shared quadcopter dynamics (continuous-time ẋ) + parameters from quad_params.json.
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "quad_examodels_dynamics.jl"))
const qp = (; (Symbol(k) => v for (k, v) in
               JSON.parsefile(joinpath(@__DIR__, "..", "quad_params.json")))...)

# Test-specific parameters (state bounds, obstacles, horizon, timestep) from test_params.json.
# JSON has no infinity, so `null` denotes an unbounded (±Inf) entry — matching the
# Python ipopt/jaxipm convention in this directory.
const _tp = JSON.parsefile(joinpath(@__DIR__, "test_params.json"))
const x_lb = [v === nothing ? -Inf : Float64(v) for v in _tp["x_lb"]]
const x_ub = [v === nothing ?  Inf : Float64(v) for v in _tp["x_ub"]]

# Problem dimensions
const nx = 13
const nu = 4
const N  = Int(_tp["N_horizon"])
const Ts = Float64(_tp["Ts"])

# Obstacle parameters from test_params.json. `r` is the base radius; the quad's
# own radius is added on (matching the Python `r__ = r_ + quad_radius`).
const quad_radius = Float64(_tp["quad_radius"])
const obs_xc = [Float64(o["xc"]) for o in _tp["obstacles"]]
const obs_yc = [Float64(o["yc"]) for o in _tp["obstacles"]]
const obs_r  = [Float64(o["r"]) + quad_radius for o in _tp["obstacles"]]

# Cost weights
const Q_cost = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Initial state
const x0_init = [4.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ─────────────────────────────────────────────────────────────────────────────
# Build ExaModel
#
# Variables layout (1-indexed, column-major):
#   X[i,k] = state i at timestep k       → var index (k-1)*nx + i,         k=1..N
#   U[j,k] = control j at timestep k     → var index N*nx + (k-1)*nu + j,  k=1..N-1
# ─────────────────────────────────────────────────────────────────────────────
function build_model(x0_ic_vec::Vector{Float64} = x0_init; backend = nothing)
    c = ExaCore(Float64; backend = backend)

    # --- Decision variables with initial guess ---
    # Build start values: linear interpolation for xyz, q0=1, hover for controls
    start_xyz = [x0_ic_vec[1], x0_ic_vec[2], x0_ic_vec[3]]
    end_xyz   = [0.0, 0.0, -2.0]
    w_hover   = 522.9847140714692

    n_x_vars = N * nx
    n_u_vars = (N - 1) * nu
    n_total  = n_x_vars + n_u_vars

    # Compute initial values and bounds for all variables
    x0_vals = zeros(n_total)
    lvar = fill(-Inf, n_total)
    uvar = fill( Inf, n_total)

    for k in 1:N
        t = (k - 1) / (N - 1)
        base = (k - 1) * nx
        # xyz interpolation
        for d in 1:3
            x0_vals[base + d] = (1 - t) * start_xyz[d] + t * end_xyz[d]
        end
        x0_vals[base + 4] = 1.0  # q0 = 1
        # state bounds
        for i in 1:nx
            lvar[base + i] = MADNLP_NO_BOUNDS ? -Inf : x_lb[i]
            uvar[base + i] = MADNLP_NO_BOUNDS ?  Inf : x_ub[i]
        end
    end
    for k in 1:(N-1)
        base = n_x_vars + (k - 1) * nu
        for j in 1:nu
            x0_vals[base + j] = w_hover
            lvar[base + j] = MADNLP_NO_BOUNDS ? -Inf : qp.minWmotor
            uvar[base + j] = MADNLP_NO_BOUNDS ?  Inf : qp.maxWmotor
        end
    end

    z = variable(c, n_total; start = x0_vals, lvar = lvar, uvar = uvar)

    # Helper to get state/control variable references
    # X[i,k] → z[(k-1)*nx + i]
    # U[j,k] → z[n_x_vars + (k-1)*nu + j]
    xi(k, i) = z[(k-1)*nx + i]
    uj(k, j) = z[n_x_vars + (k-1)*nu + j]

    # --- Objective: sum of Q[i] * X[i,k]^2 ---
    # Build data table for objective terms: (var_index, weight) for nonzero Q
    obj_data = [(k, i, Q_cost[i]) for k in 1:N for i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * xi(k, i)^2 for (k, i, w) in obj_data)

    # --- Initial condition: X[:,1] == x0_ic_vec ---
    ic_data = [(i, x0_ic_vec[i]) for i in 1:nx]
    constraint(c, xi(1, i) - x0v for (i, x0v) in ic_data; lcon = 0.0, ucon = 0.0)

    # --- Dynamics: explicit-Euler step on the shared continuous-time ẋ ---
    #     X[:,k+1] - X[:,k] - f(X[:,k], U[:,k])·Ts == 0   (one constraint per state)
    dyn_indices = [(k,) for k in 1:N-1]
    for i in 1:nx
        constraint(c, xi(k+1, i) - xi(k, i) - quad_xdot(xi, uj, k, qp)[i] * Ts
                   for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
    end

    # --- Obstacle avoidance: (x-xc)^2 + (y-yc)^2 - r^2*multiplier >= 0 ---
    obs_data = [
        (k, obs_xc[j], obs_yc[j], obs_r[j]^2 * (1.0 + (k-1)*Ts*0.1))
        for k in 1:N-1 for j in 1:length(obs_xc)
    ]
    constraint(c,
        (xi(k, 1) - xc)^2 + (xi(k, 2) - yc)^2 - r2m
        for (k, xc, yc, r2m) in obs_data; lcon = 0.0, ucon = Inf)

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
        tol                      = MADNLP_TOL,
        print_level              = print_level,
    )
    result = MadNLP.solve!(solver)
    return result
end

function x0_from_angle(theta::Float64, radius::Float64)
    return [radius * cos(theta), radius * sin(theta), 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
end

function run_sector(SECTOR_DEG::Float64, N_RUNS::Int)
    RADIUS = sqrt(4.0^2 + 4.0^2)

    # Sector centered at π/4; SECTOR_DEG controls the total arc angular width.
    #   SECTOR_DEG=90  → [0, π/2]   (original "corner" sector)
    #   SECTOR_DEG=180 → [-π/4, 3π/4]
    sector_half = deg2rad(SECTOR_DEG) / 2.0
    sector_center = pi / 4.0
    angles = collect(range(sector_center - sector_half,
                           stop = sector_center + sector_half, length = N_RUNS))

    println("Building ExaModel on GPU (CUDABackend) — JIT warmup")
    nlp_warm = build_model(x0_from_angle(angles[1], RADIUS); backend = CUDABackend())
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
    z_all    = zeros(0, 0)   # rebuttal run H: raw solution vectors (X blocks then U blocks)
    times    = zeros(Float64, N_RUNS)
    iters    = zeros(Int32, N_RUNS)
    obj_vals = fill(NaN, N_RUNS)
    success  = zeros(Bool, N_RUNS)
    starts   = zeros(Float64, N_RUNS, nx)

    println("\nRunning $N_RUNS sequential cold solves over linspaced angles in [0, π/2]")
    t0 = time()
    desc_str = "MadNLP nav"
    rep_start = zeros(Float64, MADNLP_REPEATS); rep_stop = zeros(Float64, MADNLP_REPEATS)
    rep_wall = zeros(Float64, MADNLP_REPEATS); rep_throughput = zeros(Float64, MADNLP_REPEATS)
    rep_solve_wall = zeros(Float64, MADNLP_REPEATS)   # sum of timed solve calls only (model build excluded)
    status_codes = fill(Int32(-1), N_RUNS)
    total_time = 0.0
    for rep in 1:MADNLP_REPEATS
    println("[VARIANCE] rep $rep/$MADNLP_REPEATS START epoch=$(time())")
    t0 = time()   # per-repeat wall clock (fix: was set once before the repeat loop)
    prog = Progress(N_RUNS; desc=desc_str, showspeed=true)
    for i in 1:N_RUNS
        x0_i = x0_from_angle(angles[i], RADIUS)
        starts[i, :] = x0_i

        nlp_i = build_model(x0_i; backend = CUDABackend())
        CUDA.synchronize()
        t1 = time()
        result = solve_once(nlp_i; print_level=MadNLP.ERROR)
        CUDA.synchronize()
        t2 = time()

        times[i] = t2 - t1
        iters[i] = result.iter
        status_codes[i] = Int32(Int(result.status))
        success[i] = (result.status == MadNLP.SOLVE_SUCCEEDED)
        try
            obj_vals[i] = result.objective
        catch
            try
                obj_vals[i] = result.f
            catch
            end
        end

        z_sol = Array(result.solution)
        if size(z_all, 2) == 0; z_all = fill(NaN, N_RUNS, length(z_sol)); end
        z_all[i, :] = z_sol
        if !any(isnan, z_sol)
            X_flat = z_sol[1:N*nx]
            X_all[i, :, :] = reshape(X_flat, nx, N)' |> collect
        end

        sr = sum(success[1:i]) / i
        next!(prog; showvalues=[(:succ, round(sr, digits=3))])
    end
    finish!(prog)
    total_time = time() - t0
    rep_start[rep] = t0; rep_stop[rep] = time(); rep_wall[rep] = total_time
    rep_throughput[rep] = N_RUNS / total_time
    rep_solve_wall[rep] = sum(times)
    println("[VARIANCE] rep $rep/$MADNLP_REPEATS STOP  epoch=$(time())  wall=$(round(total_time, digits=2))s  throughput=$(round(N_RUNS / total_time, digits=3)) solves/s  success=$(sum(success))/$N_RUNS")
    end  # repeats
    if MADNLP_REPEATS > 1
        println("[VARIANCE] throughput mean=$(round(mean(rep_throughput), digits=3)) std=$(round(std(rep_throughput), digits=3)) over $MADNLP_REPEATS repeats")
    end

    logs_dir = joinpath(@__DIR__, "logs")
    mkpath(logs_dir)
    out_path = joinpath(logs_dir, "madnlp_sector$(Int(SECTOR_DEG))$(MADNLP_SUFFIX)_results.npz")
    npzwrite(out_path,
             Dict("X_all" => X_all,
                  "z_all" => z_all,
                  "status_codes" => status_codes,
                  "rep_start_epoch" => rep_start, "rep_stop_epoch" => rep_stop,
                  "rep_wall" => rep_wall, "rep_throughput" => rep_throughput,
                  "rep_solve_wall" => rep_solve_wall,
                  "times" => times,
                  "iters" => iters,
                  "obj_vals" => obj_vals,
                  "success" => success,
                  "starts" => starts,
                  "angles" => angles,
                  "total_time" => [total_time],
                  "N" => [N],
                  "Ts" => [Ts],
                  "N_RUNS" => [N_RUNS],
                  "radius" => [RADIUS],
                  "sector_deg" => [SECTOR_DEG]))

    _madnlp_meta_write(out_path, Dict("N_RUNS" => N_RUNS, "solve_only_throughput" => N_RUNS ./ rep_solve_wall))
    n_succ = sum(success)
    mean_solve_ms = n_succ > 0 ? mean(times[success]) * 1000 : 0.0
    println("\nMadNLP nav: saved $out_path")
    println("MadNLP nav: total_time=$(round(total_time, digits=2))s, success=$n_succ/$N_RUNS, mean_solve=$(round(mean_solve_ms, digits=2))ms")
end

# Sweep over every init_angles sector width in test_params.json (e.g. 90°, 180°),
# saving one npz per sector — matching the Python ipopt/jaxipm loop convention.
function main()
    N_RUNS = parse(Int, get(ENV, "MADNLP_N_RUNS", string(_tp["N_RUNS_seq"])))
    for SECTOR_DEG in _tp["init_angles"]
        println("\n================ init_angle (sector) = $(SECTOR_DEG)° ================")
        run_sector(Float64(SECTOR_DEG), N_RUNS)
    end
end

main()
