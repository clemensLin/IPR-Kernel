# Generates data/raw/tfim_thermal/hs/HS Kernel L=<L>.csv
#
# Hilbert-Schmidt (fidelity) kernel between Gibbs states of the periodic
# transverse-field Ising chain
#
#     H(h) = -J sum_i Z_i Z_{i+1} - h sum_i X_i,
#
#     K_beta(h_a, h_b) = Tr[ rho_beta(h_a) rho_beta(h_b) ],
#
# with each rho normalized to unit Hilbert-Schmidt norm, Tr[rho^2] = 1, so the
# diagonal of the kernel is exactly one for every beta > 0. The beta = 0 slice is
# the untouched identity MPO and therefore carries Tr[Id^2] = 2^L instead; the
# committed data has this same offset and it is reproduced here on purpose.
#
# Boundary conditions are periodic, as for the IPR-kernel dataset. An earlier
# version of this dataset used open boundaries; it was regenerated so that both
# kernels describe the same states. Verified against exact thermal
# diagonalization at L = 10 (agreement to the Trotter error, ~2e-5).
#
# The beta grid stops at 2: periodic Gibbs MPOs reach chi ~ 120 at beta = 2
# already for L = 20, and every analysis of this kernel uses beta <= 2.
#
# The fields evolve in separate worker processes, ideally one per field:
#   julia --project generate_data_tfim_thermal_hs_kernel.jl --sizes 20 --workers 41
# Threads within one process scale poorly here (garbage-collector contention).
#
# Output per size (beta outer, then h_a, then h_b):
#     beta,h_a,h_b,kernel_real,kernel_imag
#
# Usage:
#   julia --project generate_data_tfim_thermal_hs_kernel.jl
#   julia --project generate_data_tfim_thermal_hs_kernel.jl --sizes 10,20
#
# Provenance: the main() sweep of the prepare_gibbs_ising.jl draft, since removed.

using Distributed
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using CairoMakie

# Parameters reproducing the committed raw data.
const H_VALS = collect(0.0:0.1:4.0)      # 41 points
const BETA_VALS = collect(0.0:0.1:2.0)   # 21 points
const SIZES = collect(10:10:100)
const J = 1.0
const DTAU = 0.01
const MAXDIM = 200
const CUTOFF = 1e-12

"""
    gibbs_trotter_gates(sites, N, J, h; dtau)

Symmetric Trotter gates for the periodic TFIM chain. `apply_dag=true` maps
`rho -> U rho U'`, so a forward-plus-backward sweep of gates
`exp(-dtau/4 * h_local)` advances beta by `dtau`.
"""
function gibbs_trotter_gates(sites::Vector{<:Index}, N::Int, J::Float64, h::Float64;
                             dtau::Float64=DTAU)
    gates = ITensor[]
    for j in 1:N
        s1, s2 = sites[j], sites[j == N ? 1 : j + 1]
        hj = -J * op("Z", s1) * op("Z", s2)

        # On a ring every site belongs to two bonds, so each carries half its field.
        hj += -0.5h * op("X", s1) * op("Id", s2)
        hj += -0.5h * op("Id", s1) * op("X", s2)

        push!(gates, exp(-dtau / 4.0 * hj))
    end
    append!(gates, reverse(gates))
    return gates
end

"""
    advance_beta!(rho, gates, from_beta, to_beta; dtau, maxdim, cutoff)

Evolve `rho` in imaginary time from `from_beta` to `to_beta`, returning the new
MPO and the beta actually reached. After every step the MPO is divided by its
exact Hilbert-Schmidt norm `sqrt(Tr rho^2)`, which both keeps the entries in
range and fixes the normalization convention: every returned state has
`Tr[rho^2] = 1`.
"""
function advance_beta!(rho::MPO, gates::Vector{ITensor}, from_beta::Float64, to_beta::Float64;
                       dtau::Float64=DTAU, maxdim::Int=MAXDIM, cutoff::Float64=CUTOFF)
    beta = from_beta
    while beta < to_beta - 1e-9
        rho = apply(gates, rho; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
        rho /= sqrt(real(inner(rho, rho)))
        beta += dtau
    end
    return rho, beta
end

"""
    prepare_ising_gibbs_state(sites, N, J, h, beta_vals; dtau, maxdim, cutoff)

Thermal density matrices of the periodic TFIM chain at every beta in `beta_vals`, as
MPOs. Kept for interactive use and for checking single-h results; the sweep
driver below does not call it, because materializing all betas at once does not
scale (see `sweep_size`).

`beta_vals` must be sorted ascending and its entries should be multiples of
`dtau`; the evolution is monotonic and each target beta is captured in passing.
"""
function prepare_ising_gibbs_state(sites::Vector{<:Index}, N::Int, J::Float64, h::Float64,
                                   beta_vals::Vector{Float64};
                                   dtau::Float64=DTAU, maxdim::Int=MAXDIM,
                                   cutoff::Float64=CUTOFF)
    gates = gibbs_trotter_gates(sites, N, J, h; dtau=dtau)
    rho = MPO(sites, "Id")
    beta = 0.0
    rho_list = MPO[]

    for beta_target in beta_vals
        rho, beta = advance_beta!(rho, gates, beta, Float64(beta_target);
                                  dtau=dtau, maxdim=maxdim, cutoff=cutoff)
        push!(rho_list, rho)
    end

    @printf("  h = %.2f: final beta = %.2f, max bond dim = %d\n",
            h, beta_vals[end], maxlinkdim(rho))
    return rho_list
end

# Per-process store of the Gibbs states a worker owns: field index => (rho, beta, gates).
const FIELD_STATES = Dict{Int,Tuple{MPO,Float64,Vector{ITensor}}}()

function init_field!(j::Int, sites::Vector{<:Index}, L::Int, h::Float64)
    FIELD_STATES[j] = (MPO(sites, "Id"), 0.0, gibbs_trotter_gates(sites, L, J, h))
    return nothing
end

function advance_field!(j::Int, beta::Float64)
    rho, current, gates = FIELD_STATES[j]
    rho, current = advance_beta!(rho, gates, current, beta)
    FIELD_STATES[j] = (rho, current, gates)
    return rho
end

release_fields!() = (empty!(FIELD_STATES); GC.gc(); nothing)

"""
    sweep_size(L, output_dir; h_vals, beta_vals)

Evaluate the kernel on the (beta, h_a, h_b) grid and write it to CSV.

All `length(h_vals)` Gibbs states are evolved in lockstep in beta. At each target
beta the full `h_a x h_b` block is computed and written, then the states advance
to the next beta. Only `length(h_vals)` MPOs are ever live, instead of the
`length(h_vals) * length(beta_vals)` the draft held: the kernel at one beta only
ever couples states at that same beta, so keeping every beta resident bought
nothing. Each h still sees the identical sequence of Trotter steps and
normalizations, so the results are unchanged (reproduces the committed L=10 file
to 1.9e-15 absolute).

Each field is owned by one worker process (round robin over `workers()`; without
added workers everything runs in the main process). At every beta the workers
advance their states in parallel and send them back, and the main process forms
the kernel. Entries are exact Hilbert-Schmidt inner products
`inner(rho_a, rho_b) = Tr[rho_a rho_b]` of the real symmetric MPOs, so no truncated
MPO product is formed, and the block is symmetric by construction.
"""
function sweep_size(L::Int, output_dir::String;
                    h_vals::AbstractVector{<:Real}=H_VALS,
                    beta_vals::AbstractVector{<:Real}=BETA_VALS)
    betas = collect(Float64.(beta_vals))
    fields = collect(Float64.(h_vals))
    n_h = length(fields)
    sites = siteinds("S=1/2", L)

    println("L=$L: $(n_h) fields x $(length(betas)) betas, " *
            "$(length(betas) * n_h^2) kernel entries.")

    pool = workers()
    owner = [pool[mod1(j, length(pool))] for j in 1:n_h]
    @sync for j in 1:n_h
        @async remotecall_wait(init_field!, owner[j], j, sites, L, fields[j])
    end
    states = Vector{MPO}(undef, n_h)

    mkpath(output_dir)
    filename = joinpath(output_dir, "HS Kernel L=$(L).csv")
    last_block = zeros(ComplexF64, n_h, n_h)

    open(filename, "w") do io
        println(io, "beta,h_a,h_b,kernel_real,kernel_imag")

        for beta in betas
            @sync for j in 1:n_h
                @async states[j] = remotecall_fetch(advance_field!, owner[j], j, beta)
            end

            for j1 in 1:n_h, j2 in j1:n_h
                last_block[j1, j2] = inner(states[j1], states[j2])
                last_block[j2, j1] = conj(last_block[j1, j2])
            end
            for j1 in 1:n_h, j2 in 1:n_h
                println(io, "$beta,$(fields[j1]),$(fields[j2])," *
                            "$(real(last_block[j1, j2])),$(imag(last_block[j1, j2]))")
            end
            flush(io)
            @printf("  beta = %.2f done, max bond dim = %d\n",
                    beta, maximum(maxlinkdim, states))
            flush(stdout)
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="h_a", ylabel="h_b",
              title="Hilbert-Schmidt kernel, L = $L, beta = $(betas[end])")
    hm = heatmap!(ax, fields, fields, real.(last_block))
    Colorbar(fig[1, 2], hm)
    save(joinpath(output_dir, "hs_kernel_L=$(L).png"), fig)

    @sync for w in unique(owner)
        @async remotecall_wait(release_fields!, w)
    end
    println("Wrote $filename")
    return filename
end

function parse_args(argv::Vector{String})
    opts = Dict{String,Any}("sizes" => SIZES, "outdir" => nothing, "workers" => 0)
    i = 1
    while i <= length(argv)
        key = argv[i]
        i + 1 <= length(argv) || error("$key requires a value")
        if key == "--sizes"
            opts["sizes"] = [parse(Int, s) for s in split(argv[i + 1], ',')]
        elseif key == "--outdir"
            opts["outdir"] = argv[i + 1]
        elseif key == "--workers"
            opts["workers"] = parse(Int, argv[i + 1])
        else
            error("unknown argument: $key")
        end
        i += 2
    end
    return opts
end

function main(argv::Vector{String}=String[])
    opts = parse_args(argv)
    if opts["workers"] > 0
        addprocs(opts["workers"]; exeflags=["--project=$(Base.active_project())", "-t", "1"])
        # Workers load this file for the definitions; `main` only runs as a script.
        @everywhere workers() include($(abspath(@__FILE__)))
        # One BLAS thread per process: the parallelism is over fields.
        @everywhere BLAS.set_num_threads(1)
    end

    repo_root = dirname(@__DIR__)
    default_dir = joinpath(repo_root, "data", "raw", "tfim_thermal", "hs")
    output_dir = opts["outdir"] === nothing ? default_dir : String(opts["outdir"])

    for L in opts["sizes"]
        println("\n", "="^70)
        println("HS kernel, L = $L")
        println("="^70)
        sweep_size(L, output_dir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
