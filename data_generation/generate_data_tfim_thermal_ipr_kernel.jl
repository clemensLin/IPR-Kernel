# Generates data/raw/tfim_thermal/ipr/ipr_thermal_L=<L>.csv
#
# Gibbs states of the periodic transverse-field Ising chain
#
#     H = -J sum_i Z_i Z_{i+1} - h sum_i X_i,
#
# are built as MPOs by symmetric Trotterized imaginary-time evolution from the
# identity (beta = 0). At every point of the (h, beta) grid the diagonal purity
#
#     IPR = sum_x p(x)^2,   p(x) = <x|rho|x> / Tr rho,
#
# is estimated by perfect sampling of the MPO diagonal, in the computational (Z)
# basis and after a global Hadamard rotation (X basis). Sampling p(x) and
# averaging p(x) gives an unbiased IPR estimate.
#
# Output per size (h outer loop, beta inner loop):
#     h,beta,ipr_z,ipr_x
#
# Usage:
#   julia --project generate_data_tfim_thermal_ipr_kernel.jl
#   julia --project generate_data_tfim_thermal_ipr_kernel.jl --sizes 10,20
#
# Provenance: the mc_heatmap_sweep.jl draft, with the MPO-diagonal sampler from the
# mc_ipr_ising.jl draft inlined below. Both drafts have since been removed; this
# script is self-contained.

using ITensors
using ITensorMPS
using Printf
using Random
using Base.Threads

ITensors.op(::OpName"H", ::SiteType"S=1/2") = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]

# ---------------------------------------------------------------------------
# MPO-diagonal sampler, carried over verbatim from the mc_ipr_ising.jl draft so this
# script has no dependencies beyond ITensors. Inlining it rather than `include`-ing
# also dropped a transitive dependency on HDF5 that nothing here needed. Verified
# bit-identical to the draft's version at a fixed seed.
# ---------------------------------------------------------------------------

"""
    precompute_right_env(rho, s)

Right environments of the physical trace, `R[j] = Tr_{>=j} rho`, so the sampler can
compute conditional marginals in one pass instead of re-contracting the tail.
"""
function precompute_right_env(rho::MPO, s::Vector{<:Index})
    N = length(rho)
    R = Vector{ITensor}(undef, N + 1)
    R[N + 1] = ITensor(1.0)
    for j in reverse(1:N)
        sj = s[j]
        # delta projects |st><st| onto the sum, tracing the physical index cheaply.
        TR = rho[j] * delta(sj, prime(sj))
        R[j] = TR * R[j + 1]
    end
    return R
end

"""
    precompute_projectors(s)

Diagonal projectors `|st><st|` for every site and local state, built once so the
sampling loop allocates nothing.
"""
function precompute_projectors(s::Vector{<:Index})
    N = length(s)
    projs = Vector{Vector{ITensor}}(undef, N)
    for j in 1:N
        sj = s[j]
        projs[j] = Vector{ITensor}(undef, dim(sj))
        for st in 1:dim(sj)
            p = ITensor(sj, prime(sj))
            p[sj => st, prime(sj) => st] = 1.0
            projs[j][st] = p
        end
    end
    return projs
end

"""
    fast_sample_and_eval(rho, s, R, projs)

Draw one configuration from the MPO diagonal `p(x) = <x|rho|x>` and return it
together with `p(x)`, accumulated from the conditional marginals as the sweep
proceeds so no second pass is needed.
"""
function fast_sample_and_eval(rho::MPO, s::Vector{<:Index}, R::Vector{ITensor},
                              projs::Vector{Vector{ITensor}})
    N = length(rho)
    L = ITensor(1.0)
    config = zeros(Int, N)
    current_px = 1.0

    for j in 1:N
        sj = s[j]
        A = rho[j]
        LA = L * A
        LAR = LA * R[j + 1]

        P = zeros(Float64, dim(sj))
        for state in 1:dim(sj)
            P[state] = real(LAR[sj => state, prime(sj) => state])
        end

        P .= max.(P, 0.0)
        norm_factor = sum(P)
        if norm_factor > 0
            P ./= norm_factor
        else
            P .= 1.0 / dim(sj)
        end

        r = rand()
        chosen_state = findfirst(x -> r <= x, cumsum(P))
        if chosen_state === nothing
            chosen_state = dim(sj)
        end
        config[j] = chosen_state

        current_px *= P[chosen_state]
        L = LA * projs[j][chosen_state]
    end

    return config, current_px
end

"""
    estimate_ipr_mc(rho, s; num_samples)

Monte Carlo estimate of `IPR = sum_x p(x)^2` for the normalized MPO diagonal.
Sampling `x` from `p` and averaging `p(x)` is unbiased, so no reweighting is needed.
Returns the normalized IPR, the unnormalized one, and `Tr rho`.

Note the accumulation is over threads via `Threads.atomic_add!`, so the summation
order varies between runs and the result moves by an ulp or two even at a fixed
seed. That is ~1e-16 relative, fourteen orders below the sampling noise.
"""
function estimate_ipr_mc(rho::MPO, s::Vector{<:Index}; num_samples::Int=2000)
    Z = real(tr(rho))
    rho_norm = rho / Z

    R_env = precompute_right_env(rho_norm, s)
    projs = precompute_projectors(s)

    ipr_sum = Threads.Atomic{Float64}(0.0)
    Threads.@threads for i in 1:num_samples
        _, p_val = fast_sample_and_eval(rho_norm, s, R_env, projs)
        Threads.atomic_add!(ipr_sum, p_val)
    end

    mc_ipr = ipr_sum[] / num_samples
    return mc_ipr, mc_ipr * Z, Z
end

# Parameters reproducing the committed raw data.
const H_VALS = collect(0.0:0.1:4.0)      # 41 points
const BETA_VALS = collect(0.0:0.1:6.0)   # 61 points
const SIZES = collect(10:10:100)
const NUM_SAMPLES = 2000
const J = 1.0
const DTAU = 0.01
const MAXDIM = 400
const CUTOFF = 1e-8

"""
    trotter_gates(sites, N, J, h, dtau)

Symmetric second-order Trotter gates for `exp(-dtau * H / 4)` on a periodic
chain, mirrored so that one forward-plus-backward sweep advances beta by `dtau`.

`4 Sz Sz` is the Pauli `Z Z`; each site receives the transverse field from both
of its bonds, so the per-bond weight `-h Sx` sums to `-h X` per site.
"""
function trotter_gates(sites, N::Int, J::Float64, h::Float64, dtau::Float64)
    gates = ITensor[]
    for j in 1:N
        s1, s2 = sites[j], sites[j == N ? 1 : j + 1]
        hj = -J * 4.0 * op("Sz", s1) * op("Sz", s2)
        hj += -h * op("Sx", s1) * op("Id", s2)
        hj += -h * op("Id", s1) * op("Sx", s2)
        push!(gates, exp(-dtau / 4.0 * hj))
    end
    append!(gates, reverse(gates))
    return gates
end

"""
    sweep_size(N, output_dir; h_vals, beta_vals, num_samples)

Sweep the (h, beta) grid for one chain length and stream the results to CSV.
The MPO is evolved monotonically in beta, so each h column costs one evolution
to `maximum(beta_vals)` rather than one per beta.
"""
function sweep_size(N::Int, output_dir::String;
                    h_vals::AbstractVector{<:Real}=H_VALS,
                    beta_vals::AbstractVector{<:Real}=BETA_VALS,
                    num_samples::Int=NUM_SAMPLES)
    sites = siteinds("S=1/2", N)

    println("Starting thermal IPR sweep for N=$N.")
    println("Grid: $(length(h_vals)) h x $(length(beta_vals)) beta = " *
            "$(length(h_vals) * length(beta_vals)) evaluations")

    mkpath(output_dir)
    filename = joinpath(output_dir, "ipr_thermal_L=$(N).csv")

    open(filename, "w") do io
        println(io, "h,beta,ipr_z,ipr_x")

        for h in h_vals
            @printf("=> Sweeping h = %.2f\n", h)
            rho = MPO(sites, "Id")
            gates = trotter_gates(sites, N, J, Float64(h), DTAU)
            gates_H = [op("H", sites[j]) for j in 1:N]

            current_beta = 0.0
            for target_beta in beta_vals
                while current_beta < target_beta - 1e-6
                    rho = apply(gates, rho; cutoff=CUTOFF, maxdim=MAXDIM, apply_dag=true)
                    current_beta += DTAU
                end
                rho /= tr(rho)

                ipr_z, _, _ = estimate_ipr_mc(rho, sites; num_samples=num_samples)
                rho_X = apply(gates_H, rho; apply_dag=true)
                ipr_x, _, _ = estimate_ipr_mc(rho_X, sites; num_samples=num_samples)

                println(io, "$h,$target_beta,$ipr_z,$ipr_x")
                flush(io)

                if target_beta == last(beta_vals)
                    println("   max bond dim at h=$h, beta=$target_beta: $(maxlinkdim(rho))")
                end
            end
        end
    end

    println("Wrote $filename")
    return filename
end

function parse_args(argv::Vector{String})
    opts = Dict{String,Any}("sizes" => SIZES, "seed" => 1234, "outdir" => nothing)
    i = 1
    while i <= length(argv)
        key = argv[i]
        i + 1 <= length(argv) || error("$key requires a value")
        if key == "--sizes"
            opts["sizes"] = [parse(Int, s) for s in split(argv[i + 1], ',')]
        elseif key == "--seed"
            opts["seed"] = parse(Int, argv[i + 1])
        elseif key == "--outdir"
            opts["outdir"] = argv[i + 1]
        else
            error("unknown argument: $key")
        end
        i += 2
    end
    return opts
end

function main(argv::Vector{String}=String[])
    opts = parse_args(argv)
    Random.seed!(opts["seed"])

    repo_root = dirname(@__DIR__)
    default_dir = joinpath(repo_root, "data", "raw", "tfim_thermal", "ipr")
    output_dir = opts["outdir"] === nothing ? default_dir : String(opts["outdir"])

    for N in opts["sizes"]
        println("\n", "="^70)
        sweep_size(N, output_dir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
