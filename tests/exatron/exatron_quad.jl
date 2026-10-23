# ExaTron (trust-region Newton, bound constraints only) on the paper's three
# quadrotor NLPs, via slacks + augmented Lagrangian (tests/exatron/alm_tron.jl).
#
#   julia tests/exatron/exatron_quad.jl nav   [90|180]        # sector width (deg)
#   julia tests/exatron/exatron_quad.jl track [1.0|1.5|2.0]   # avg_vel
#   julia tests/exatron/exatron_quad.jl multi [2|4]           # N_quads
#
# Without a variant every value from the task's test_params.json is run in turn.
# Instances are identical to the MadNLP drivers (same angles / same Python
# sampler stream / same rotation pool), so logs/madnlp_*_results.npz from the
# sibling test directories serve as the reference objective per instance.
#
#   EXATRON_N_RUNS=10        instances per variant (default: N_RUNS_seq of the task)
#   EXATRON_TIME_LIMIT=300   seconds per solve
#   EXATRON_MAX_OUTER=50     ALM outer iterations
#   EXATRON_OUT_SUFFIX=_x    appended to the results file name
#
# Results: tests/exatron/logs/exatron_<task>_<variant>[suffix]_results.npz (+ .meta.json)
#
# NOTE: this is ExaTron's sparse (Julia) TRON on the full problem. ExaTron's
# batched GPU kernel holds a dense n x n Hessian per thread block in shared
# memory and reduces with warp shuffles, so it requires n <= 32 variables per
# problem; the quadrotor NLPs have 506 to 2200 variables and cannot be run
# through it without an ADMM-style decomposition (see README.md).

using NPZ, JSON, Printf, Statistics, LinearAlgebra, NLPModels
include(joinpath(@__DIR__, "quad_models.jl"))
include(joinpath(@__DIR__, "alm_tron.jl"))
cd(REPO_ROOT)

const TASK = length(ARGS) >= 1 ? ARGS[1] : "nav"
const VARIANT = length(ARGS) >= 2 ? ARGS[2] : nothing
const SUFFIX = get(ENV, "EXATRON_OUT_SUFFIX", "")
const TIME_LIMIT = parse(Float64, get(ENV, "EXATRON_TIME_LIMIT", "300"))
const MAX_OUTER = parse(Int, get(ENV, "EXATRON_MAX_OUTER", "50"))
const LOGS = joinpath(@__DIR__, "logs"); mkpath(LOGS)

if TASK == "track"
    get!(ENV, "JULIA_CONDAPKG_BACKEND", "Null")
    get!(ENV, "JULIA_PYTHONCALL_EXE", "/opt/Miniconda/bin/python3")
    using PythonCall
    pyimport("sys").path.insert(0, REPO_ROOT)
end

const STATUS_CODE = Dict(:Converged => 0, :MaxOuter => 1, :TimeLimit => 2, :NonFinite => 3)

function load_ref(path)
    isfile(path) || return nothing
    d = npzread(path)
    return d
end

function meta_write(out_path, extra)
    git(args...) = try strip(read(Cmd(["git", "-C", @__DIR__, args...]), String)) catch; "unknown" end
    d = Dict{String,Any}("task" => TASK, "variant" => VARIANT, "time_limit" => TIME_LIMIT, "max_outer" => MAX_OUTER,
        "exatron_version" => string(pkgversion(ExaTron)), "julia" => string(VERSION),
        "jaxipm_release_commit" => git("rev-parse", "HEAD"), "dirty" => git("status", "--porcelain") != "",
        "hostname" => gethostname(), "argv" => ARGS, "script" => @__FILE__,
        "status_legend" => Dict(string(v) => string(k) for (k, v) in STATUS_CODE),
        "alm" => "slacks on every inequality row; method of multipliers (mu0=10, x10, mu_max=1e8, eta=1/mu^0.1 schedule as ExaAdmm); inner solver ExaTron sparse TRON (Julia), frtol=fatol=0, gtol=1e-6, ctol=1e-6")
    merge!(d, extra)
    open(out_path * ".meta.json", "w") do io; JSON.print(io, d, 2); end
end

# ── generic instance loop ────────────────────────────────────────────────────
# build(i) -> (nlp, extra::Dict) ; unpack(z) -> X_all slice ; ok(z, extra) -> task success flag
function run_variant(tag, n_runs, build, unpack, ok, ref_obj, Xshape)
    println("\n================ ExaTron $TASK $tag : $n_runs instances ================")
    nlp1, _ = build(1)
    W1 = ALMWork(nlp1)
    println("  NLP: nvar=$(nlp1.meta.nvar) ncon=$(nlp1.meta.ncon) (equalities $(nlp1.meta.ncon - length(W1.ineq)), inequalities->slacks $(length(W1.ineq)))")
    println("  bound-constrained ALM problem: n=$(W1.nw) variables, Hessian nnz (tril) $(length(W1.pat_rows))")
    println("  warmup (JIT) ...")
    t_jit = @elapsed alm_tron_solve(nlp1; opts = ALMOptions(time_limit = TIME_LIMIT, max_outer = MAX_OUTER))
    println("  warmup solve incl. JIT: $(round(t_jit, digits = 1)) s")

    obj_vals = fill(NaN, n_runs); times = zeros(n_runs); outer = zeros(Int32, n_runs); minor = zeros(Int32, n_runs)
    cnorm = fill(NaN, n_runs); pgnorm = fill(NaN, n_runs); status = fill(Int32(-1), n_runs)
    nfev = zeros(Int32, n_runs); mu_final = fill(NaN, n_runs); task_ok = zeros(Bool, n_runs)
    X_all = fill(NaN, (n_runs, Xshape...)); z_all = zeros(0, 0)
    t0_all = time()
    for i in 1:n_runs
        nlp, extra = build(i)
        res = alm_tron_solve(nlp; opts = ALMOptions(time_limit = TIME_LIMIT, max_outer = MAX_OUTER))
        obj_vals[i] = res.f; times[i] = res.time; outer[i] = res.outer; minor[i] = res.minor
        cnorm[i] = res.cnorm; pgnorm[i] = res.pgnorm; status[i] = STATUS_CODE[res.status]; nfev[i] = res.nfev
        mu_final[i] = res.mu
        if size(z_all, 2) == 0; z_all = fill(NaN, n_runs, length(res.z)); end
        z_all[i, :] = res.z
        X_all[i, ntuple(_ -> Colon(), length(Xshape))...] = unpack(res.z)
        task_ok[i] = res.status == :Converged && ok(res.z, extra)
        rel = ref_obj === nothing ? NaN : abs(res.f - ref_obj[i]) / max(1.0, abs(ref_obj[i]))
        @printf("  [%3d/%d] %-10s f=%.6f  ref=%s  relgap=%.1e  |r|=%.1e pg=%.1e  outer=%2d  tron_it=%4d  mu=%.0e  %.1fs%s\n",
                i, n_runs, res.status, res.f, ref_obj === nothing ? "   n/a" : @sprintf("%.6f", ref_obj[i]), rel,
                res.cnorm, res.pgnorm, res.outer, res.minor, res.mu, res.time, task_ok[i] ? "" : "  <-- not ok")
        flush(stdout)
    end
    total = time() - t0_all
    conv = status .== 0
    relgap = ref_obj === nothing ? fill(NaN, n_runs) : abs.(obj_vals .- ref_obj) ./ max.(1.0, abs.(ref_obj))
    @printf("\n  summary %s: converged %d/%d, task-ok %d/%d, median time %.1fs (mean %.1fs), median TRON iters %d, median rel obj gap vs MadNLP %.1e, max %.1e\n",
            tag, sum(conv), n_runs, sum(task_ok), n_runs, median(times), mean(times), Int(round(median(minor))),
            (ref_obj === nothing || !any(conv)) ? NaN : median(relgap[conv]), (ref_obj === nothing || !any(conv)) ? NaN : maximum(relgap[conv]))
    out_path = joinpath(LOGS, "exatron_$(TASK)_$(tag)$(SUFFIX)_results.npz")
    npzwrite(out_path, Dict("obj_vals" => obj_vals, "ref_obj_madnlp" => ref_obj === nothing ? fill(NaN, n_runs) : ref_obj,
                            "rel_obj_gap" => relgap, "times" => times, "outer_iters" => outer, "tron_iters" => minor,
                            "cnorm" => cnorm, "pgnorm" => pgnorm, "status_codes" => status, "success" => conv,
                            "task_ok" => task_ok, "nfev" => nfev, "mu_final" => mu_final, "X_all" => X_all, "z_all" => z_all,
                            "total_time" => [total], "N_RUNS" => [n_runs]))
    meta_write(out_path, Dict("n_converged" => sum(conv), "n_task_ok" => sum(task_ok), "median_time_s" => median(times),
                              "nvar_nlp" => nlp1.meta.nvar, "ncon_nlp" => nlp1.meta.ncon, "n_alm_vars" => W1.nw,
                              "n_slacks" => length(W1.ineq), "warmup_s" => t_jit))
    println("  saved $out_path")
end

unpack_single(z, N) = collect(reshape(z[1:N*nx], nx, N)')

if TASK == "nav"
    sectors = VARIANT === nothing ? NAV.init_angles : [parse(Float64, VARIANT)]
    n_runs = parse(Int, get(ENV, "EXATRON_N_RUNS", string(NAV.N_RUNS_seq)))
    for sec in sectors
        angles = nav_angles(sec, NAV.N_RUNS_seq)
        ref = load_ref(joinpath(REPO_ROOT, "tests", "quad_nav_circle", "logs", "madnlp_sector$(Int(sec))_results.npz"))
        ref_obj = ref === nothing ? nothing : ref["obj_vals"][1:n_runs]
        build = i -> (build_nav(nav_x0_from_angle(angles[i])), Dict())
        run_variant("sector$(Int(sec))", n_runs, build, z -> unpack_single(z, NAV.N), (z, e) -> true, ref_obj, (NAV.N, nx))
    end
elseif TASK == "track"
    vels = VARIANT === nothing ? TRACK.avg_vels : [parse(Float64, VARIANT)]
    n_runs = parse(Int, get(ENV, "EXATRON_N_RUNS", string(TRACK.N_RUNS_seq)))
    for v in vels
        all_x0, all_xr, all_s, _ = track_random_inits(TRACK.N_RUNS_seq, v; delta_std = 0.5, seed = 0)
        ref = load_ref(joinpath(REPO_ROOT, "tests", "quad_track_avoid", "logs", "madnlp_v$(round(v, digits = 1))_results.npz"))
        ref_obj = ref === nothing ? nothing : ref["obj_vals"][1:n_runs]
        if ref !== nothing && haskey(ref, "all_s")
            @assert maximum(abs.(ref["all_s"][1:n_runs] .- all_s[1:n_runs])) < 1e-12 "instance mismatch vs MadNLP reference"
        end
        build = i -> (build_track(all_x0[i, :], all_xr[i, :, :]), Dict())
        run_variant("v$(round(v, digits = 1))", n_runs, build, z -> unpack_single(z, TRACK.N), (z, e) -> true, ref_obj, (TRACK.N, nx))
    end
elseif TASK == "multi"
    nqs = VARIANT === nothing ? MULTI.N_quads : [parse(Int, VARIANT)]
    n_runs = parse(Int, get(ENV, "EXATRON_N_RUNS", string(MULTI.N_RUNS_seq)))
    for Nq in nqs
        ref = load_ref(joinpath(REPO_ROOT, "tests", "quad_multi_swap", "logs", "madnlp_$(Nq)_results.npz"))
        ref_obj = ref === nothing ? nothing : ref["obj_vals"][1:n_runs]
        function build(i)
            k_i = (i - 1) % MULTI.n_pool
            nlp, starts, goals = build_multi(Nq, MULTI.R; phi = pool_phi(k_i), psi = pool_psi(k_i))
            return nlp, Dict("goals" => goals)
        end
        stride_x = MULTI.N * nx
        unpack = z -> begin
            X = fill(NaN, Nq, MULTI.N, nx)
            for q in 1:Nq
                X[q, :, :] = reshape(z[(q-1)*stride_x + 1 : q*stride_x], nx, MULTI.N)'
            end
            X
        end
        ok = (z, e) -> multi_check_success(unpack(z), e["goals"])
        run_variant("$(Nq)quads", n_runs, build, unpack, ok, ref_obj, (Nq, MULTI.N, nx))
    end
else
    error("unknown task $TASK (nav|track|multi)")
end
