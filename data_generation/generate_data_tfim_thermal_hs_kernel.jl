# Generates Data/Raw/Data TFIM Thermal/Data HS Kernel/HS Kernel L=<L>.csv
#
# Hilbert-Schmidt (fidelity) kernel between Gibbs states of the open-boundary
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
# Note the boundary conditions differ from the IPR-kernel dataset: this one is
# open (N - 1 bonds), the IPR kernel is periodic. Both were verified against
# exact thermal diagonalization at L = 10.
#
# Output per size (beta outer, then h_a, then h_b):
#     beta,h_a,h_b,kernel_real,kernel_imag
#
# Usage:
#   julia --project generate_data_tfim_thermal_hs_kernel.jl
#   julia --project generate_data_tfim_thermal_hs_kernel.jl --sizes 10,20
#
# Provenance: the main() sweep of the prepare_gibbs_ising.jl draft, since removed.

using ITensors
using ITensorMPS
using Printf
using CairoMakie

# Parameters reproducing the committed raw data.
const H_VALS = collect(0.0:0.1:4.0)      # 41 points
const BETA_VALS = collect(0.0:0.1:6.0)   # 61 points
const SIZES = collect(10:10:100)
const J = 1.0
const DTAU = 0.01
const MAXDIM = 200
const CUTOFF = 1e-12
const KERNEL_MAXDIM = 800
const KERNEL_CUTOFF = 1e-12

"""
    gibbs_trotter_gates(sites, N, J, h; dtau)

Symmetric Trotter gates for the open TFIM chain. `apply_dag=true` maps
`rho -> U rho U'`, so a forward-plus-backward sweep of gates
`exp(-dtau/4 * h_local)` advances beta by `dtau`.
"""
function gibbs_trotter_gates(sites::Vector{<:Index}, N::Int, J::Float64, h::Float64;
                             dtau::Float64=DTAU)
    gates = ITensor[]
    for j in 1:(N - 1)
        s1, s2 = sites[j], sites[j + 1]
        hj = -J * op("Z", s1) * op("Z", s2)

        # Interior sites appear in two bonds, so their field weight is halved.
        hj += (j == 1 ? -h : -h * 0.5) * op("X", s1) * op("Id", s2)
        hj += (j == N - 1 ? -h : -h * 0.5) * op("Id", s1) * op("X", s2)

        push!(gates, exp(-dtau / 4.0 * hj))
    end
    append!(gates, reverse(gates))
    return gates
end

"""
    advance_beta!(rho, gates, from_beta, to_beta; dtau, maxdim, cutoff)

Evolve `rho` in imaginary time from `from_beta` to `to_beta`, returning the new
MPO and the beta actually reached. After every step the MPO is divided by
`sqrt(Tr rho^2)`, which both keeps the entries in range and fixes the
normalization convention: every returned state has `Tr[rho^2] = 1`.
"""
function advance_beta!(rho::MPO, gates::Vector{ITensor}, from_beta::Float64, to_beta::Float64;
                       dtau::Float64=DTAU, maxdim::Int=MAXDIM, cutoff::Float64=CUTOFF)
    beta = from_beta
    while beta < to_beta - 1e-9
        rho = apply(gates, rho; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
        rho /= sqrt(tr(apply(rho, rho; cutoff=cutoff, maxdim=maxdim)))
        beta += dtau
    end
    return rho, beta
end

"""
    prepare_ising_gibbs_state(sites, N, J, h, beta_vals; dtau, maxdim, cutoff)

Thermal density matrices of the open TFIM chain at every beta in `beta_vals`, as
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

Measured bond dimensions stay well under `MAXDIM` (chi = 14 at L=40, beta=2),
so the draft's version was memory-hungry rather than infeasible; this form is
preferred because it is strictly cheaper, not because the other could not run.
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

    gates = [gibbs_trotter_gates(sites, L, J, h) for h in fields]
    states = [MPO(sites, "Id") for _ in fields]
    current = zeros(Float64, n_h)

    mkpath(output_dir)
    filename = joinpath(output_dir, "HS Kernel L=$(L).csv")
    last_block = zeros(ComplexF64, n_h, n_h)

    open(filename, "w") do io
        println(io, "beta,h_a,h_b,kernel_real,kernel_imag")

        for beta in betas
            for j in 1:n_h
                states[j], current[j] = advance_beta!(states[j], gates[j], current[j], beta)
            end

            for j1 in 1:n_h, j2 in 1:n_h
                last_block[j1, j2] = tr(apply(states[j1], states[j2];
                                              maxdim=KERNEL_MAXDIM, cutoff=KERNEL_CUTOFF))
                println(io, "$beta,$(fields[j1]),$(fields[j2])," *
                            "$(real(last_block[j1, j2])),$(imag(last_block[j1, j2]))")
            end
            flush(io)
            @printf("  beta = %.2f done, max bond dim = %d\n",
                    beta, maximum(maxlinkdim, states))
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="h_a", ylabel="h_b",
              title="Hilbert-Schmidt kernel, L = $L, beta = $(betas[end])")
    hm = heatmap!(ax, fields, fields, real.(last_block))
    Colorbar(fig[1, 2], hm)
    save(joinpath(output_dir, "hs_kernel_L=$(L).png"), fig)

    println("Wrote $filename")
    return filename
end

function parse_args(argv::Vector{String})
    opts = Dict{String,Any}("sizes" => SIZES, "outdir" => nothing)
    i = 1
    while i <= length(argv)
        key = argv[i]
        i + 1 <= length(argv) || error("$key requires a value")
        if key == "--sizes"
            opts["sizes"] = [parse(Int, s) for s in split(argv[i + 1], ',')]
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

    repo_root = dirname(@__DIR__)
    default_dir = joinpath(repo_root, "Data", "Raw", "Data TFIM Thermal", "Data HS Kernel")
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
