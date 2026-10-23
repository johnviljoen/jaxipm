# Augmented-Lagrangian wrapper that lets ExaTron (a bound-constrained
# trust-region Newton solver) attack a general NLP
#
#     min f(z)  s.t.  lcon <= g(z) <= ucon,  lvar <= z <= uvar
#
# in the way ExaAdmm does for its ACOPF components:
#   * every inequality row g_i(z) in [l_i, u_i] gets a slack s_i with the row
#     turned into the equality g_i(z) - s_i = 0 and the bound l_i <= s_i <= u_i;
#   * all equalities r(w) = 0 (w = [z; s]) are handled by the method of
#     multipliers: minimise L_A(w) = f(z) + λᵀr + (μ/2)‖r‖² over the box with
#     TRON, then update λ ← λ + μ r when ‖r‖∞ <= η, else μ ← 10 μ
#     (schedule copied from ExaAdmm's auglag_linelimit kernel: μ0 = 10,
#     η = 1/μ^0.1, ω = 1/μ, η ← η/μ^0.9 and ω ← ω/μ after an accepted update,
#     μ_max = 1e8 and 50 outer iterations as in ExaAdmm's defaults; η is floored
#     at the final feasibility tolerance so a feasible iterate always triggers a
#     multiplier update rather than a penalty increase).
#
# f, ∇f, g, J and the Lagrangian Hessian come from any NLPModels model (here an
# ExaModel); the ALM Hessian H_L(z, λ+μr) + μ JᵀJ (plus the slack cross terms)
# is assembled as a sparse matrix with a fixed pattern and handed to TRON's
# sparse incomplete-Cholesky-preconditioned CG.

using ExaTron, NLPModels, SparseArrays, LinearAlgebra, Printf

struct ALMOptions
    mu0::Float64
    mu_max::Float64
    mu_factor::Float64
    ctol::Float64          # final ‖r‖∞ tolerance
    gtol::Float64          # final projected-gradient tolerance (TRON's "tol")
    max_outer::Int
    max_minor::Int         # TRON iterations per inner solve
    max_feval::Int
    time_limit::Float64
    verbose::Int
end
ALMOptions(; mu0 = 10.0, mu_max = 1e8, mu_factor = 10.0, ctol = 1e-6, gtol = 1e-6,
             max_outer = 50, max_minor = 500, max_feval = 2000, time_limit = 300.0, verbose = 0) =
    ALMOptions(mu0, mu_max, mu_factor, ctol, gtol, max_outer, max_minor, max_feval, time_limit, verbose)

mutable struct ALMWork
    nlp
    n::Int                 # original variables
    m::Int                 # constraints
    ineq::Vector{Int}      # constraint rows that get a slack
    slack_of::Vector{Int}  # row -> slack index (0 for equality rows)
    nw::Int                # n + length(ineq)
    lw::Vector{Float64}
    uw::Vector{Float64}
    # NLP evaluation buffers
    gz::Vector{Float64}
    gval::Vector{Float64}
    r::Vector{Float64}
    y::Vector{Float64}
    jrows::Vector{Int}
    jcols::Vector{Int}
    jvals::Vector{Float64}
    hrows::Vector{Int}
    hcols::Vector{Int}
    hvals::Vector{Float64}
    # ALM Hessian assembly: fixed COO atomic-term lists, assembled into a CSC each time
    Iv::Vector{Int}
    Jv::Vector{Int}
    Vv::Vector{Float64}
    pair_a::Vector{Int}    # JᵀJ terms: jac entry a, jac entry b, weight
    pair_b::Vector{Int}
    pair_w::Vector{Float64}
    n_hL::Int
    n_pair::Int
    n_cross::Int
    cross_a::Vector{Int}   # jac entries in inequality rows (−μ J_Iᵀ cross terms)
    pat_rows::Vector{Int}
    pat_cols::Vector{Int}
    # multipliers / penalty
    lambda::Vector{Float64}
    mu::Float64
    # counters
    nfev::Int
    ngev::Int
    nhev::Int
    t_eval::Float64
    zcache::Vector{Float64}   # last z for which gval/jvals were computed (gradient+hessian)
end

function ALMWork(nlp)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    lcon, ucon = nlp.meta.lcon, nlp.meta.ucon
    ineq = [i for i in 1:m if lcon[i] != ucon[i]]
    slack_of = zeros(Int, m)
    for (j, i) in enumerate(ineq); slack_of[i] = j; end
    nw = n + length(ineq)
    lw = vcat(copy(nlp.meta.lvar), lcon[ineq]); uw = vcat(copy(nlp.meta.uvar), ucon[ineq])
    jrows, jcols = jac_structure(nlp)
    hrows, hcols = hess_structure(nlp)
    nnzj, nnzh = length(jrows), length(hrows)
    # group jac entries by row for the JᵀJ pair expansion
    byrow = [Int[] for _ in 1:m]
    for a in 1:nnzj; push!(byrow[jrows[a]], a); end
    pair_a = Int[]; pair_b = Int[]; pair_w = Float64[]
    for i in 1:m
        ents = byrow[i]
        for (ia, a) in enumerate(ents), b in ents[ia:end]
            push!(pair_a, a); push!(pair_b, b)
            push!(pair_w, a == b ? 1.0 : (jcols[a] == jcols[b] ? 2.0 : 1.0))
        end
    end
    cross_a = [a for a in 1:nnzj if slack_of[jrows[a]] > 0]
    # atomic COO lists (lower triangle: row >= col)
    Iv = Int[]; Jv = Int[]
    for k in 1:nnzh
        push!(Iv, max(hrows[k], hcols[k])); push!(Jv, min(hrows[k], hcols[k]))
    end
    for p in 1:length(pair_a)
        ca, cb = jcols[pair_a[p]], jcols[pair_b[p]]
        push!(Iv, max(ca, cb)); push!(Jv, min(ca, cb))
    end
    for a in cross_a
        push!(Iv, n + slack_of[jrows[a]]); push!(Jv, jcols[a])
    end
    for j in 1:length(ineq)
        push!(Iv, n + j); push!(Jv, n + j)
    end
    pat = sparse(Iv, Jv, ones(length(Iv)), nw, nw)
    pr, pc, _ = findnz(pat)
    return ALMWork(nlp, n, m, ineq, slack_of, nw, lw, uw,
                   zeros(n), zeros(m), zeros(m), zeros(m),
                   jrows, jcols, zeros(nnzj), hrows, hcols, zeros(nnzh),
                   Iv, Jv, zeros(length(Iv)), pair_a, pair_b, pair_w,
                   nnzh, length(pair_a), length(cross_a), cross_a, pr, pc,
                   zeros(m), 10.0, 0, 0, 0, 0.0, fill(NaN, n))
end

# residual r(w): g_i(z) − lcon_i (equalities) or g_i(z) − s_i (inequalities)
function residual!(W::ALMWork, w::AbstractVector)
    z = view(w, 1:W.n)
    cons!(W.nlp, z, W.gval)
    lcon = W.nlp.meta.lcon
    @inbounds for i in 1:W.m
        j = W.slack_of[i]
        W.r[i] = j == 0 ? W.gval[i] - lcon[i] : W.gval[i] - w[W.n + j]
    end
    return W.r
end

function alm_f(W::ALMWork, w::AbstractVector)
    t0 = time()
    residual!(W, w)
    fz = obj(W.nlp, view(w, 1:W.n))
    W.nfev += 1
    W.t_eval += time() - t0
    return fz + dot(W.lambda, W.r) + 0.5 * W.mu * dot(W.r, W.r)
end

function alm_grad!(W::ALMWork, w::AbstractVector, gw::AbstractVector)
    t0 = time()
    z = view(w, 1:W.n)
    residual!(W, w)
    @. W.y = W.lambda + W.mu * W.r
    grad!(W.nlp, z, W.gz)
    jac_coord!(W.nlp, z, W.jvals)
    fill!(gw, 0.0)
    @inbounds for i in 1:W.n; gw[i] = W.gz[i]; end
    @inbounds for a in 1:length(W.jvals)
        gw[W.jcols[a]] += W.jvals[a] * W.y[W.jrows[a]]
    end
    @inbounds for (j, i) in enumerate(W.ineq)
        gw[W.n + j] = -W.y[i]
    end
    W.ngev += 1
    W.t_eval += time() - t0
    return gw
end

# ALM Hessian values in the fixed CSC pattern order (lower triangle)
function alm_hess_values!(W::ALMWork, w::AbstractVector, values::AbstractVector)
    t0 = time()
    z = view(w, 1:W.n)
    residual!(W, w)
    @. W.y = W.lambda + W.mu * W.r
    jac_coord!(W.nlp, z, W.jvals)
    hess_coord!(W.nlp, z, W.y, W.hvals; obj_weight = 1.0)
    mu = W.mu
    V = W.Vv
    @inbounds for k in 1:W.n_hL; V[k] = W.hvals[k]; end
    off = W.n_hL
    @inbounds for p in 1:W.n_pair
        V[off + p] = mu * W.pair_w[p] * W.jvals[W.pair_a[p]] * W.jvals[W.pair_b[p]]
    end
    off += W.n_pair
    @inbounds for (q, a) in enumerate(W.cross_a)
        V[off + q] = -mu * W.jvals[a]
    end
    off += W.n_cross
    @inbounds for j in 1:length(W.ineq); V[off + j] = mu; end
    H = sparse(W.Iv, W.Jv, V, W.nw, W.nw)
    nz = nonzeros(H)
    @assert length(nz) == length(values)
    copyto!(values, nz)
    W.nhev += 1
    W.t_eval += time() - t0
    return values
end

function max_violation(W::ALMWork, w::AbstractVector)
    residual!(W, w)
    return maximum(abs, W.r; init = 0.0)
end

"""
    alm_tron_solve(nlp; w0 = nothing, opts = ALMOptions())

Run the slack + augmented-Lagrangian scheme with ExaTron as the inner solver.
Returns a NamedTuple with the final z, objective, status, counters.
"""
function alm_tron_solve(nlp; w0 = nothing, opts::ALMOptions = ALMOptions())
    t_start = time()
    W = ALMWork(nlp)
    n, nw = W.n, W.nw
    # start: NLP x0 plus slacks set to the (clipped) constraint values
    if w0 === nothing
        z0 = copy(nlp.meta.x0)
        cons!(nlp, z0, W.gval)
        s0 = [clamp(W.gval[i], nlp.meta.lcon[i], nlp.meta.ucon[i]) for i in W.ineq]
        w0 = vcat(z0, s0)
    end
    eval_f(x) = alm_f(W, x)
    eval_grad_f(x, g) = alm_grad!(W, x, g)
    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for k in 1:length(W.pat_rows)
                rows[k] = W.pat_rows[k]; cols[k] = W.pat_cols[k]
            end
        else
            alm_hess_values!(W, x, values)
        end
    end
    prob = ExaTron.createProblem(nw, W.lw, W.uw, length(W.pat_rows), eval_f, eval_grad_f, eval_h;
                                 max_minor = opts.max_minor, max_feval = opts.max_feval,
                                 frtol = 0.0, fatol = 0.0)   # stop on the projected gradient only
    prob.x .= w0
    W.mu = opts.mu0
    fill!(W.lambda, 0.0)
    mu = W.mu
    eta = 1.0 / mu^0.1
    omega = 1.0 / mu
    status = :MaxOuter
    outer = 0
    minor_total = 0
    cnorm = Inf; pgnorm = Inf
    hist = NamedTuple[]
    while outer < opts.max_outer
        outer += 1
        ExaTron.setOption(prob, "tol", max(omega, opts.gtol))
        tron_status = ExaTron.solveProblem(prob)
        minor_total += prob.minor_iter
        cnorm = max_violation(W, prob.x)
        pgnorm = prob.gnorm_inf
        push!(hist, (outer = outer, mu = mu, eta = eta, cnorm = cnorm, pg = pgnorm, f = obj(nlp, view(prob.x, 1:n)),
                     tron = tron_status, minor = prob.minor_iter))
        opts.verbose > 0 && @printf("  ALM %3d  mu=%.1e eta=%.1e  |r|=%.2e  pg=%.2e  f=%.6f  tron=%s (%d it)\n",
                                    outer, mu, eta, cnorm, pgnorm, hist[end].f, tron_status, prob.minor_iter)
        if cnorm <= max(eta, opts.ctol)
            if cnorm <= opts.ctol && pgnorm <= opts.gtol
                status = :Converged; break
            end
            residual!(W, prob.x)
            @. W.lambda += mu * W.r
            eta = eta / mu^0.9
            omega = omega / mu
        else
            mu = min(opts.mu_max, mu * opts.mu_factor)
            W.mu = mu
            eta = 1.0 / mu^0.1
            omega = 1.0 / mu
        end
        if time() - t_start > opts.time_limit
            status = :TimeLimit; break
        end
        if !all(isfinite, prob.x)
            status = :NonFinite; break
        end
    end
    z = prob.x[1:n]
    fz = obj(nlp, z)
    return (z = z, w = copy(prob.x), f = fz, status = status, outer = outer, minor = minor_total,
            cnorm = cnorm, pgnorm = pgnorm, mu = mu, nfev = W.nfev, ngev = W.ngev, nhev = W.nhev,
            time = time() - t_start, t_eval = W.t_eval, hist = hist, lambda = copy(W.lambda),
            nw = nw, nnz_hess = length(W.pat_rows))
end
