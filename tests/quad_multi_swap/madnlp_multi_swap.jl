
# MadNLP GPU solver for the multi-quadcopter rendezvous problem.
#
# Parametric on N_QUADS via env var (default 2). Layout matches the CasADi /
# JAX / cusadi_jaxipm variants in this directory:
#
#   z = [X_0.flat, X_1.flat, ..., X_{Nq-1}.flat,
#        U_0.flat, U_1.flat, ..., U_{Nq-1}.flat]
#
# Per-quad X_q is (N, 13) row-major and U_q is (N-1, 4) row-major.
#
# Cost: sum over quads and stages of Q_cost .* (x_q[k] - goal_q).^2.
# Goal is the diametrically opposite point on the circle (non-zero pos).
#
# Equality constraints: initial condition + Euler dynamics per quad.
# Inequality: pairwise ||pos_i[k] - pos_j[k]||^2 - MIN_DIST^2 >= 0 for
# all pairs i<j and stages k=1..N-1. No per-stage margin multiplier.
#
# Result file: logs/madnlp_{N_quads}_results.npz to match the casadi/jaxipm
# result schema so analyze_results.py can load all three.

ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "1")

using ExaModels, MadNLP, MadNLPGPU, CUDA, NPZ, ProgressMeter, Statistics, JSON
using LinearAlgebra

CUDA.allowscalar(false)
println("Using GPU: ", CUDA.device())

# ─────────────────────────────────────────────────────────────────────────────
# Shared quadcopter dynamics (continuous-time ẋ) + parameters from quad_params.json.
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "quad_examodels_dynamics.jl"))
const qp = (; (Symbol(k) => v for (k, v) in
               JSON.parsefile(joinpath(@__DIR__, "..", "quad_params.json")))...)

# Test-specific parameters (horizon, timestep, geometry, collision, bounds) from
# test_params.json. State bounds are parametric in R, so the R-independent pieces
# (xy margin beyond R, z bound, vel/rate bound) are stored and combined in code.
const _tp = JSON.parsefile(joinpath(@__DIR__, "test_params.json"))

# Problem dimensions
const nx = 13
const nu = 4
const N  = Int(_tp["N_horizon"])
const Ts = Float64(_tp["Ts"])

# State-bound pieces (combined with R per-quad inside build_model).
const XY_MARGIN      = Float64(_tp["xy_margin"])
const Z_BOUND        = Float64(_tp["z_bound"])
const VEL_RATE_BOUND = Float64(_tp["vel_rate_bound"])

# Collision parameters (keep in sync with casadi/jaxipm versions).
const QUAD_RADIUS = Float64(_tp["quad_radius"])
const SAFETY      = Float64(_tp["safety"])
# Doubled from (2*r_quad + safety) = 0.5 to 1.0 for a more substantial
# avoidance cushion between quads.
const MIN_DIST    = 2.0 * (2 * QUAD_RADIUS + SAFETY)   # 1.0 m centre-centre
const MIN_DIST_SQ = MIN_DIST^2

# Cost weights: position 1, quaternion unweighted, velocity 0.1 (relaxed),
# angular rate 1.
const Q_cost = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0]

# Hover control value.
const w_hover = 522.9847140714692

# ─────────────────────────────────────────────────────────────────────────────
# Problem geometry (per-quad starts/goals on a circle of radius R).
# ─────────────────────────────────────────────────────────────────────────────
function quad_start(q::Int, Nq::Int, R::Float64)
    θ = 2π * (q - 1) / Nq
    # position (3) + identity quaternion (4) + zero velocity (3) + zero rate (3)
    return [R*cos(θ), R*sin(θ), 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end

function quad_goal(q::Int, Nq::Int, R::Float64)
    θ = 2π * (q - 1) / Nq + π
    return [R*cos(θ), R*sin(θ), 0.0]
end

# ─────────────────────────────────────────────────────────────────────────────
# Build ExaModel
#
# Per-quad offsets:
#   stride_x = N * nx
#   stride_u = (N-1) * nu
#   X_q[i,k]  → z[(q-1)*stride_x + (k-1)*nx + i]     q=1..Nq, k=1..N, i=1..nx
#   U_q[j,k]  → z[Nq*stride_x + (q-1)*stride_u + (k-1)*nu + j]
# ─────────────────────────────────────────────────────────────────────────────
function build_model(Nq::Int, R::Float64; backend = nothing)
    c = ExaCore(Float64; backend = backend)

    stride_x = N * nx
    stride_u = (N - 1) * nu
    n_x_vars = Nq * stride_x
    n_u_vars = Nq * stride_u
    n_total  = n_x_vars + n_u_vars

    # State bounds: x,y ∈ ±(R+XY_MARGIN); z ∈ ±Z_BOUND; q free; vel & rate ∈ ±VEL_RATE_BOUND.
    vrb = VEL_RATE_BOUND
    x_lb = [-(R+XY_MARGIN), -(R+XY_MARGIN), -Z_BOUND, -Inf, -Inf, -Inf, -Inf,
            -vrb, -vrb, -vrb, -vrb, -vrb, -vrb]
    x_ub = [ (R+XY_MARGIN),  (R+XY_MARGIN),  Z_BOUND,  Inf,  Inf,  Inf,  Inf,
             vrb,  vrb,  vrb,  vrb,  vrb,  vrb]

    # Hover cold guess for every quad (feasible since starts are spaced
    # 2 R sin(π/Nq) apart, which exceeds MIN_DIST for any sensible R).
    starts = [quad_start(q, Nq, R) for q in 1:Nq]
    goals  = [quad_goal(q, Nq, R)  for q in 1:Nq]

    x0_vals = zeros(n_total)
    lvar = fill(-Inf, n_total)
    uvar = fill( Inf, n_total)

    for q in 1:Nq
        base_q = (q - 1) * stride_x
        for k in 1:N
            base_k = base_q + (k - 1) * nx
            for i in 1:nx
                x0_vals[base_k + i] = starts[q][i]
                lvar[base_k + i]    = x_lb[i]
                uvar[base_k + i]    = x_ub[i]
            end
        end
    end
    for q in 1:Nq
        base_q = n_x_vars + (q - 1) * stride_u
        for k in 1:(N-1)
            base_k = base_q + (k - 1) * nu
            for j in 1:nu
                x0_vals[base_k + j] = w_hover
                lvar[base_k + j]    = qp.minWmotor
                uvar[base_k + j]    = qp.maxWmotor
            end
        end
    end

    z = variable(c, n_total; start = x0_vals, lvar = lvar, uvar = uvar)

    xi(q, k, i) = z[(q-1)*stride_x + (k-1)*nx + i]
    uj(q, k, j) = z[n_x_vars + (q-1)*stride_u + (k-1)*nu + j]

    # --- Objective: sum_q sum_k sum_i Q_cost[i] * (X_q[i,k] - goal_q[i])^2 ---
    # goal_q[i] is zero outside position slots (i=1,2,3); Q_cost[4..7]=0 already
    # zeroes out the quaternion contribution. Velocity/rate targets are zero.
    function goal_val(q, i)
        if i == 1 || i == 2 || i == 3
            return goals[q][i]
        else
            return 0.0
        end
    end
    obj_data = [(q, k, i, Q_cost[i], goal_val(q, i))
                for q in 1:Nq, k in 1:N, i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * (xi(q, k, i) - gval)^2 for (q, k, i, w, gval) in obj_data)

    # --- Initial condition per quad: X_q[:,1] == starts[q] ---
    ic_data = [(q, i, starts[q][i]) for q in 1:Nq, i in 1:nx]
    constraint(c, xi(q, 1, i) - x0v for (q, i, x0v) in ic_data; lcon = 0.0, ucon = 0.0)

    # --- Dynamics: explicit-Euler step on the shared continuous-time ẋ, per quad ---
    #     X_q[:,k+1] - X_q[:,k] - f(X_q[:,k], U_q[:,k])·Ts == 0   (one per state)
    dyn_indices = [(k,) for k in 1:N-1]
    for q in 1:Nq
        xi_q = (k, i) -> xi(q, k, i)
        uj_q = (k, j) -> uj(q, k, j)
        for i in 1:nx
            constraint(c, xi_q(k+1, i) - xi_q(k, i) - quad_xdot(xi_q, uj_q, k, qp)[i] * Ts
                       for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
        end
    end

    # --- Pairwise collision avoidance (no per-stage margin multiplier) ---
    # For every pair i<j and stage k=1..N-1: ||pos_i[k] - pos_j[k]||^2 >= MIN_DIST^2
    if Nq >= 2
        pair_data = [(i, j, k)
                     for i in 1:Nq for j in (i+1):Nq for k in 1:(N-1)]
        constraint(c,
            (xi(i, k, 1) - xi(j, k, 1))^2
            + (xi(i, k, 2) - xi(j, k, 2))^2
            + (xi(i, k, 3) - xi(j, k, 3))^2
            - MIN_DIST_SQ
            for (i, j, k) in pair_data; lcon = 0.0, ucon = Inf)
    end

    return ExaModel(c), starts, goals
end

# ─────────────────────────────────────────────────────────────────────────────
# Solve helpers
# ─────────────────────────────────────────────────────────────────────────────
function solve_once(nlp; print_level=MadNLP.INFO)
    solver = MadNLPSolver(nlp;
        kkt_system               = MadNLP.SparseCondensedKKTSystem,
        linear_solver            = MadNLPGPU.CUDSSSolver,
        cudss_algorithm          = MadNLP.CHOLESKY,
        equality_treatment       = MadNLP.RelaxEquality,
        fixed_variable_treatment = MadNLP.RelaxBound,
        dual_initialized         = true,
        tol                      = 1e-8,
        print_level              = print_level,
    )
    result = MadNLP.solve!(solver)
    return result
end

# Real success check matching casadi_multi_swap.check_success semantics.
# Returns Bool.
function check_success(X_quad_traj::Array{Float64,3}, goals::Vector{Vector{Float64}},
                       iter_count::Int, max_iter::Int;
                       goal_tol::Float64 = 0.3,
                       collision_tol::Float64 = 0.8 * MIN_DIST)
    Nq = size(X_quad_traj, 1)
    # Goal check: max end-pos distance across quads must be < goal_tol.
    goal_errs = [norm(X_quad_traj[q, end, 1:3] .- goals[q]) for q in 1:Nq]
    goal_ok = maximum(goal_errs) < goal_tol

    # Pairwise min distance over all stages.
    min_d = Inf
    for i in 1:Nq
        for j in (i+1):Nq
            for k in 1:size(X_quad_traj, 2)
                d = norm(X_quad_traj[i, k, 1:3] .- X_quad_traj[j, k, 1:3])
                if d < min_d
                    min_d = d
                end
            end
        end
    end
    no_collision = min_d > collision_tol

    iters_ok = 0 <= iter_count < max_iter
    return goal_ok && no_collision && iters_ok
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
function run_nquads(Nq::Int, R, N_RUNS::Int, max_it::Int)
    println("MadNLP multi-swap: N_quads=$Nq, N=$N, R=$R, N_RUNS=$N_RUNS")

    println("Building ExaModel on GPU (CUDABackend) — JIT warmup")
    nlp_warm, _, _ = build_model(Nq, R; backend = CUDABackend())
    println("\nProblem dimensions:")
    println("  Variables:   $(nlp_warm.meta.nvar)")
    println("  Constraints: $(nlp_warm.meta.ncon)")

    CUDA.synchronize()
    t_jit = @elapsed begin
        _ = solve_once(nlp_warm; print_level=MadNLP.ERROR)
        CUDA.synchronize()
    end
    println("Warmup solve (incl. JIT): $(round(t_jit * 1000, digits=1)) ms")

    X_all    = fill(NaN, N_RUNS, Nq, N, nx)
    times    = zeros(Float64, N_RUNS)
    iters    = zeros(Int32, N_RUNS)
    obj_vals = fill(NaN, N_RUNS)
    success  = zeros(Bool, N_RUNS)

    # Cache starts/goals from a single model build (deterministic of Nq, R).
    _, starts_vec, goals_vec = build_model(Nq, R; backend = CUDABackend())
    starts_arr = reduce(vcat, [reshape(starts_vec[q], 1, nx) for q in 1:Nq])
    goals_arr  = reduce(vcat, [reshape(goals_vec[q],  1, 3)  for q in 1:Nq])

    println("\nRunning $N_RUNS sequential cold solves")
    t0 = time()
    prog = Progress(N_RUNS; desc="MadNLP multi-swap", showspeed=true)
    for i in 1:N_RUNS
        nlp_i, _, _ = build_model(Nq, R; backend = CUDABackend())
        CUDA.synchronize()
        t1 = time()
        result = solve_once(nlp_i; print_level=MadNLP.ERROR)
        CUDA.synchronize()
        t2 = time()

        times[i] = t2 - t1
        iters[i] = result.iter
        try
            obj_vals[i] = result.objective
        catch
            try
                obj_vals[i] = result.f
            catch
            end
        end

        # Reshape solution into per-quad trajectories (Nq, N, nx).
        z_sol = Array(result.solution)
        X_quad = fill(NaN, Nq, N, nx)
        stride_x_int = N * nx
        if !any(isnan, z_sol)
            for q in 1:Nq
                X_flat = z_sol[(q-1)*stride_x_int + 1 : q*stride_x_int]
                # row-major: reshape nx × N then transpose so quad[k,i] = flat[(k-1)*nx + i]
                X_quad[q, :, :] = reshape(X_flat, nx, N)' |> collect
            end
            X_all[i, :, :, :] = X_quad
        end

        # Real success: goal reached + no collision + not at max iter.
        converged = !any(isnan, z_sol) &&
                    check_success(X_quad, goals_vec, Int(iters[i]), max_it)
        success[i] = converged

        sr = sum(success[1:i]) / i
        next!(prog; showvalues=[(:succ, round(sr, digits=3))])
    end
    finish!(prog)
    total_time = time() - t0

    logs_dir = joinpath(@__DIR__, "logs")
    mkpath(logs_dir)
    out_path = joinpath(logs_dir, "madnlp_$(Nq)_results.npz")
    npzwrite(out_path,
             Dict("X_all"      => X_all,
                  "times"      => times,
                  "iters"      => iters,
                  "obj_vals"   => obj_vals,
                  "success"    => success,
                  "total_time" => [total_time],
                  "N"          => [N],
                  "Ts"         => [Ts],
                  "N_RUNS"     => [N_RUNS],
                  "N_quads"    => [Nq],
                  "R"          => [R],
                  "starts"     => starts_arr,
                  "goals"      => goals_arr))

    n_succ = sum(success)
    mean_solve_ms = n_succ > 0 ? mean(times[success]) * 1000 : 0.0
    mean_iter_val = n_succ > 0 ? mean(iters[success]) : NaN
    println("\nMadNLP multi-swap: saved $out_path")
    println("MadNLP multi-swap: total_time=$(round(total_time, digits=2))s, " *
            "success=$n_succ/$N_RUNS, " *
            "mean_solve=$(round(mean_solve_ms, digits=2))ms, " *
            "mean_iters=$(round(mean_iter_val, digits=1))")
end

# Sweep over every N_quads in test_params.json (e.g. 2, 4), saving one npz per
# value — matching the Python ipopt/jaxipm loop convention.
function main()
    R      = _tp["R"]
    N_RUNS = Int(_tp["N_RUNS_seq"])
    max_it = 500
    for Nq in _tp["N_quads"]
        println("\n================ N_quads = $Nq ================")
        run_nquads(Int(Nq), R, N_RUNS, max_it)
    end
end

main()
