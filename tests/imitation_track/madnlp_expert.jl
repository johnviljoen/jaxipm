# MadNLP (GPU, cuDSS) receding-horizon expert server for the track-avoid DAgger
# pipeline (tests/imitation_track). Same NLP as the paper's MadNLP driver
# (tests/quad_track_avoid/madnlp_track_avoid.jl): identical dynamics include,
# bounds, cost, quaternion band and obstacle constraints -- but the measured
# state, the reference horizon and the obstacle centres over the horizon are
# ExaModels PARAMETERS (closed loop needs absolute time), and the planner knobs
# of the jaxipm expert (obstacle margin, first constrained stage, input cost)
# are reproduced so all three experts solve the identical NLP.
#
# One ExaModel is built once; per instance the parameters and the warm start
# (nlp.meta.x0) are updated and a FRESH MadNLPSolver is constructed and solved
# with the paper's options (full IPM initialisation per solve, as in the paper
# driver; only the model build is amortised).
#
# Protocol (driven by tests/imitation_track/expert_madnlp.py):
#   argv: margin k_start r_u tol
#   stdout: "READY" after the JIT warmup, then "DONE <out.npz>" per request
#   stdin : one request path per line ("QUIT" to exit); request npz holds
#           x0 (P,13), xr (P,N,13), obs (P,N,N_OBS,2), warm (P,nz)
#   output: <path>.out.npz with Z (P,nz) (NaN rows for failures), status (P,)
#           MadNLP status codes, iters (P,), obj (P,), times (P,) per-solve
#           wall, elapsed (1,) wall of the whole loop

using ExaModels, MadNLP, MadNLPGPU, CUDA, NPZ, JSON
CUDA.allowscalar(false)

const MARGIN  = parse(Float64, ARGS[1])
const K_START = parse(Int, ARGS[2])
const R_U     = parse(Float64, ARGS[3])
const TOL     = parse(Float64, ARGS[4])
const MU_INIT = parse(Float64, get(ENV, "MADNLP_MU_INIT", "1.0"))   # paper driver: 1.0

include(joinpath(@__DIR__, "..", "quad_examodels_dynamics.jl"))
const qp = (; (Symbol(k) => v for (k, v) in
               JSON.parsefile(joinpath(@__DIR__, "..", "quad_params.json")))...)
const _tp = JSON.parsefile(joinpath(@__DIR__, "..", "quad_track_avoid", "test_params.json"))
const x_lb = [v === nothing ? -Inf : Float64(v) for v in _tp["x_lb"]]
const x_ub = [v === nothing ?  Inf : Float64(v) for v in _tp["x_ub"]]
const nx = 13
const nu = 4
const N  = Int(_tp["N_horizon"])
const Ts = Float64(_tp["Ts"])
const Q_cost = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
const OBS_R = [Float64(o["r"]) for o in _tp["obstacles"]]
const N_OBS = length(OBS_R)
const w_hover = 522.9847140714692
const n_x_vars = N * nx
const n_u_vars = (N - 1) * nu
const nz = n_x_vars + n_u_vars

struct TrackModel
    core
    nlp
    θx0
    θxr
    θobs
end

function build_param_model(; backend = CUDABackend())
    c = ExaCore(Float64; backend = backend)
    x0_vals = zeros(nz); lvar = fill(-Inf, nz); uvar = fill(Inf, nz)
    for k in 1:N
        base = (k - 1) * nx
        x0_vals[base + 4] = 1.0
        for i in 1:nx
            lvar[base + i] = x_lb[i]; uvar[base + i] = x_ub[i]
        end
    end
    for k in 1:(N-1)
        base = n_x_vars + (k - 1) * nu
        for j in 1:nu
            x0_vals[base + j] = w_hover
            lvar[base + j] = qp.minWmotor; uvar[base + j] = qp.maxWmotor
        end
    end
    z = variable(c, nz; start = x0_vals, lvar = lvar, uvar = uvar)
    θx0  = parameter(c, zeros(nx))
    θxr  = parameter(c, zeros(N * nx))           # stage-major: (k-1)*nx + i
    θobs = parameter(c, zeros(N * N_OBS * 2))    # ((k-1)*N_OBS + j-1)*2 + {1: xc, 2: yc}

    xi(k, i) = z[(k-1)*nx + i]
    uj(k, j) = z[n_x_vars + (k-1)*nu + j]

    obj_data = [(k, i, Q_cost[i]) for k in 1:N for i in 1:nx if Q_cost[i] != 0.0]
    objective(c, w * (xi(k, i) - θxr[(k-1)*nx + i])^2 for (k, i, w) in obj_data)
    if R_U > 0.0
        u_data = [(k, j) for k in 1:(N-1) for j in 1:nu]
        objective(c, R_U * (uj(k, j) - w_hover)^2 for (k, j) in u_data)
    end
    constraint(c, xi(1, i) - θx0[i] for i in 1:nx; lcon = 0.0, ucon = 0.0)
    dyn_indices = [(k,) for k in 1:N-1]
    for i in 1:nx
        constraint(c, xi(k+1, i) - xi(k, i) - quad_xdot(xi, uj, k, qp)[i] * Ts
                   for (k,) in dyn_indices; lcon = 0.0, ucon = 0.0)
    end
    quat_data = [(k,) for k in 1:N]
    constraint(c, xi(k,4)^2 + xi(k,5)^2 + xi(k,6)^2 + xi(k,7)^2
               for (k,) in quat_data; lcon = 0.5, ucon = 1.5)
    obs_data = [(k, ((k-1)*N_OBS + (j-1))*2, (OBS_R[j] + MARGIN)^2)
                for k in (K_START+1):N for j in 1:N_OBS]
    constraint(c, (xi(k, 1) - θobs[o + 1])^2 + (xi(k, 2) - θobs[o + 2])^2 - r2
               for (k, o, r2) in obs_data; lcon = 0.0, ucon = Inf)
    return TrackModel(c, ExaModel(c), θx0, θxr, θobs)
end

function make_solver(nlp; print_level = MadNLP.ERROR)
    MadNLPSolver(nlp;
        kkt_system               = MadNLP.SparseCondensedKKTSystem,
        linear_solver            = MadNLPGPU.CUDSSSolver,
        cudss_algorithm          = MadNLP.CHOLESKY,
        equality_treatment       = MadNLP.RelaxEquality,
        fixed_variable_treatment = MadNLP.RelaxBound,
        dual_initialized         = true,
        tol                      = TOL,
        mu_init                  = MU_INIT,
        first_hessian_perturbation = 1e-2,
        print_level              = print_level)
end

function solve_instance!(m::TrackModel, x0, xr_flat, obs_flat, warm)
    set_parameter!(m.core, m.θx0, CuArray(x0))
    set_parameter!(m.core, m.θxr, CuArray(xr_flat))
    set_parameter!(m.core, m.θobs, CuArray(obs_flat))
    copyto!(m.nlp.meta.x0, warm)
    CUDA.synchronize()
    t1 = time()
    solver = make_solver(m.nlp)
    res = MadNLP.solve!(solver)
    CUDA.synchronize()
    t2 = time()
    return res, t2 - t1
end

function serve(m::TrackModel, path::String)
    req = npzread(path)
    x0 = req["x0"]; xr = req["xr"]; obs = req["obs"]; warm = req["warm"]
    P = size(x0, 1)
    Z = fill(NaN, P, nz); status = fill(Int32(-99), P); iters = zeros(Int32, P)
    obj = fill(NaN, P); times = zeros(P)
    t0 = time()
    for i in 1:P
        xr_i  = vec(permutedims(xr[i, :, :], (2, 1)))          # (N,nx) -> stage-major flat
        obs_i = vec(permutedims(obs[i, :, :, :], (3, 2, 1)))   # (N,N_OBS,2) -> ((k,j),xy) flat
        res, dt = solve_instance!(m, x0[i, :], xr_i, obs_i, warm[i, :])
        status[i] = Int32(Int(res.status)); iters[i] = Int32(res.iter); times[i] = dt
        obj[i] = try res.objective catch; NaN end
        if res.status == MadNLP.SOLVE_SUCCEEDED || res.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
            Z[i, :] = Array(res.solution)
        end
        if i % 1000 == 0
            println(stderr, "    [madnlp] $i/$P solved, $(round(time() - t0, digits=1))s, success so far $(count(s -> s == 1 || s == 2, status[1:i]))"); flush(stderr)
        end
    end
    elapsed = time() - t0
    hist = Dict(string(st) => count(==(Int32(Int(st))), status) for st in instances(MadNLP.Status) if any(==(Int32(Int(st))), status))
    println(stderr, "    [madnlp] $P instances in $(round(elapsed, digits=1))s, iters mean $(round(sum(iters)/P, digits=1)), status $hist"); flush(stderr)
    out = path * ".out.npz"
    npzwrite(out, Dict("Z" => Z, "status" => status, "iters" => iters, "obj" => obj,
                       "times" => times, "elapsed" => [elapsed]))
    return out
end

function main()
    println(stderr, "[madnlp] GPU $(CUDA.device())  margin=$MARGIN k_start=$K_START r_u=$R_U tol=$TOL mu_init=$MU_INIT"); flush(stderr)
    t0 = time()
    m = build_param_model()
    # JIT warmup on a reference-like instance
    x0 = zeros(nx); x0[1] = 4.0; x0[4] = 1.0
    xr = zeros(N, nx); xr[:, 1] .= 4.0; xr[:, 4] .= 1.0
    obs = zeros(N, N_OBS, 2); obs[:, :, 1] .= [3.0 -3.0 0.0]; obs[:, :, 2] .= [3.0 2.0 -3.0]
    warm = vcat(vec(permutedims(xr, (2, 1))), fill(w_hover, n_u_vars))
    res, dt = solve_instance!(m, x0, vec(permutedims(xr, (2, 1))), vec(permutedims(obs, (3, 2, 1))), warm)
    println(stderr, "[madnlp] nvar=$(m.nlp.meta.nvar) ncon=$(m.nlp.meta.ncon) warmup status=$(res.status) iters=$(res.iter) build+JIT $(round(time() - t0, digits=1))s"); flush(stderr)
    println("READY"); flush(stdout)
    for line in eachline(stdin)
        path = strip(line)
        isempty(path) && continue
        path == "QUIT" && break
        out = serve(m, String(path))
        println("DONE ", out); flush(stdout)
    end
end

main()
