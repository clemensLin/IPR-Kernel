# Generates data/raw/tfim_{1d,2d}/L=<Lx>x<Ly>/ (only the 1D data is used in the paper)
#
# Ground states of the transverse-field Ising model on a periodic chain (Ly = 1)
# or a periodic square lattice (Ly = Lx),
#
#     H = -J sum_<ij> Z_i Z_j - h sum_i X_i,
#
# are obtained by two-site DMRG on a snake-ordered MPS. For each h the inverse
# participation ratio (IPR) of the ground state is estimated by perfect sampling
# of |psi(x)|^2 in the computational (Z) basis, in the X basis, and along a
# uniform sweep of single-site Ry(theta) rotations interpolating between them.
#
# Outputs per size, matching the committed raw data:
#   hs.csv         transverse fields
#   ipr_z.csv      IPR in the Z basis
#   ipr_x.csv      IPR in the X basis
#   all_mx.csv     <sum_i X_i> (complex, as written by DelimitedFiles)
#   new_iprs.csv   IPR after Ry(theta), one row per angle, one column per h
#   ipr_vs_h.png   overview plot
#
# Usage:
#   julia --project generate_data_tfim.jl                  # all committed sizes
#   julia --project generate_data_tfim.jl --geometry 1d --sizes 10,20
#
# Provenance: assembled from the TFIM branch of the original 2D_DMRG.jl scratch
# file, which has since been removed from the repository.

using ITensors, ITensorMPS
using DelimitedFiles
using CairoMakie
using LinearAlgebra
using Random
using Statistics
using Printf

# Hadamard and Ry are not registered for the S=1/2 site type, only for "Qubit".
ITensors.op(::OpName"H", ::SiteType"S=1/2") = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]

function ITensors.op(::OpName"Ry", ::SiteType"S=1/2"; θ::Real)
    return [cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)]
end

"""
    snake_index(x, y, Lx, Ly)

Linear MPS site index of lattice site `(x, y)` under boustrophedon ("snake")
ordering, which keeps nearest neighbours of the square lattice at MPS distance
at most `Lx`.
"""
function snake_index(x::Int, y::Int, Lx::Int, Ly::Int)
    xprime = isodd(y) ? x : Lx - x + 1
    return (y - 1) * Lx + xprime
end

"""
    build_tfim_mpo(Lx, Ly; J, h, boundary, sites)

MPO for `H = -J sum_<ij> Z_i Z_j - h sum_i X_i` on an `Lx * Ly` lattice in snake
order. With `boundary = "PBC"` a wrap-around bond is added along any direction of
extent greater than two, so `Ly = 1` gives a periodic chain and `Ly = Lx` a torus.
"""
function build_tfim_mpo(Lx::Int, Ly::Int; J::Float64=1.0, h::Float64=3.04438,
                        boundary::String="PBC", sites=nothing)
    N = Lx * Ly
    sites = sites === nothing ? siteinds("S=1/2", N) : sites

    ampo = OpSum()
    for y in 1:Ly, x in 1:Lx
        i = snake_index(x, y, Lx, Ly)

        if x < Lx
            ampo += -J, "Z", i, "Z", snake_index(x + 1, y, Lx, Ly)
        elseif boundary == "PBC" && Lx > 2
            ampo += -J, "Z", i, "Z", snake_index(1, y, Lx, Ly)
        end

        if y < Ly
            ampo += -J, "Z", i, "Z", snake_index(x, y + 1, Lx, Ly)
        elseif boundary == "PBC" && Ly > 2
            ampo += -J, "Z", i, "Z", snake_index(x, 1, Lx, Ly)
        end

        ampo += -h, "X", i
    end

    return MPO(ampo, sites), sites
end

"""
    run_dmrg(parameters; verbose)

Two-site DMRG from a random MPS with a uniform bond-dimension schedule.
"""
function run_dmrg(parameters::Dict{String,Any}; verbose::Bool=true)
    H, sites = build_tfim_mpo(parameters["Lx"], parameters["Ly"];
                              J=parameters["J"], h=parameters["h"],
                              boundary=parameters["boundary_conditions"])

    psi0 = random_mps(sites; linkdims=10)

    nsweeps = parameters["nsweeps"]
    sweeps = Sweeps(nsweeps)
    maxdim!(sweeps, fill(parameters["maxdim"], nsweeps)...)
    cutoff!(sweeps, parameters["cutoff"])
    noise!(sweeps, 1e-8)

    if verbose
        @printf("  DMRG: %dx%d, h = %.4f, maxdim = %d, cutoff = %.1e\n",
                parameters["Lx"], parameters["Ly"], parameters["h"],
                parameters["maxdim"], parameters["cutoff"])
    end

    energy, psi = dmrg(H, psi0, sweeps; outputlevel=verbose ? 1 : 0)
    return energy, psi, H, sites
end

"""
    get_amplitude(psi, config)

Amplitude `<config|psi>` for a single computational basis configuration.
"""
function get_amplitude(psi::MPS, config::Vector{Int})
    s = siteinds(psi)
    val = ITensor(1.0)
    for j in 1:length(psi)
        val *= psi[j] * dag(onehot(s[j] => config[j]))
    end
    return scalar(val)
end

"""
    compute_IPR(psi; num_samples)

Monte Carlo estimate of `IPR = sum_x |psi(x)|^4`. Perfect sampling draws `x` from
`|psi(x)|^2`, so the sample mean of `|psi(x)|^2` is an unbiased estimator of the
IPR and needs no normalisation of the sampling weights.
"""
function compute_IPR(psi::MPS; num_samples::Int=10_000)
    psi = orthogonalize(psi, 1)
    normalize!(psi)

    vals = Vector{Float64}(undef, num_samples)
    for k in 1:num_samples
        vals[k] = abs2(get_amplitude(psi, sample(psi)))
    end
    return mean(vals)
end

"""
    make_M_MPO(sites; direction, staggered)

MPO for `sum_i s_i O_i` with `O` the Pauli operator named by `direction` and
`s_i` alternating in sign when `staggered` is true.
"""
function make_M_MPO(sites; direction::String="X", staggered::Bool=false)
    os = OpSum()
    for j in 1:length(sites)
        coeff = staggered ? (isodd(j) ? 1.0 : -1.0) : 1.0
        os += coeff, direction, j
    end
    return MPO(os, sites)
end

"""
    apply_ry(psi, sites, theta)

Single-site rotation `Ry(theta)^{⊗N}` applied to `psi`. `theta = 0` leaves the
computational basis unchanged, `theta = pi/2` maps it onto the X basis.
"""
function apply_ry(psi::MPS, sites, theta::Real)
    psi_new = deepcopy(psi)
    for k in 1:length(sites)
        psi_new[k] = noprime(op("Ry", sites[k]; θ=theta) * psi_new[k])
    end
    orthogonalize!(psi_new, 1)
    normalize!(psi_new)
    return psi_new
end

"""
    sweep_size(Lx, Ly, parameters, h_vals, angle_array, output_root)

Run the full transverse-field sweep for one lattice size and write the raw CSVs.
"""
function sweep_size(Lx::Int, Ly::Int, parameters::Dict{String,Any},
                    h_vals::AbstractVector{<:Real},
                    angle_array::Union{Nothing,AbstractVector{<:Real}},
                    output_root::String)
    parameters = copy(parameters)
    parameters["Lx"] = Lx
    parameters["Ly"] = Ly

    IPR_Z = Float64[]
    IPR_X = Float64[]
    all_M = ComplexF64[]
    new_IPRs = angle_array === nothing ? nothing :
               Array{Float64,2}(undef, length(angle_array), length(h_vals))

    for (i, h) in enumerate(h_vals)
        parameters["h"] = Float64(h)
        energy, psi, H, sites = run_dmrg(parameters; verbose=true)
        orthogonalize!(psi, 1)
        normalize!(psi)

        # V-score reports how well DMRG converged; only printed, never stored.
        var_E = inner(psi', apply(H, H), psi) - energy^2
        @printf("  h = %.2f: E = %.8f, Var = %.3e, V-score = %.3e\n",
                h, energy, var_E, var_E * length(sites) / energy^2)

        ipr_z = compute_IPR(psi; num_samples=parameters["num_samples"])
        push!(IPR_Z, ipr_z)

        psi_x = apply(MPO(sites, "H"), psi)
        orthogonalize!(psi_x, 1)
        normalize!(psi_x)
        ipr_x = compute_IPR(psi_x; num_samples=parameters["num_samples"])
        push!(IPR_X, ipr_x)

        push!(all_M, inner(psi', make_M_MPO(sites; direction="X", staggered=false), psi))

        @printf("  h = %.2f: IPR_Z = %.6e, IPR_X = %.6e, <Mx> = %s\n",
                h, ipr_z, ipr_x, string(all_M[end]))

        if angle_array !== nothing
            for (j, theta) in enumerate(angle_array)
                new_IPRs[j, i] = compute_IPR(apply_ry(psi, sites, theta);
                                             num_samples=parameters["num_samples"])
            end
        end
    end

    folder = joinpath(output_root, "L=$(Lx)x$(Ly)")
    mkpath(folder)

    writedlm(joinpath(folder, "hs.csv"), collect(h_vals), ',')
    writedlm(joinpath(folder, "ipr_z.csv"), IPR_Z, ',')
    writedlm(joinpath(folder, "ipr_x.csv"), IPR_X, ',')
    writedlm(joinpath(folder, "all_mx.csv"), all_M, ',')
    if new_IPRs !== nothing
        writedlm(joinpath(folder, "new_iprs.csv"), new_IPRs, ',')
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; yscale=log10, xlabel="Transverse field h", ylabel="IPR",
              title="Inverse participation ratio vs transverse field, $(Lx)x$(Ly)")
    lines!(ax, h_vals, IPR_Z; color=:blue, label="IPR in Z basis")
    lines!(ax, h_vals, IPR_X; color=:red, label="IPR in X basis")
    if new_IPRs !== nothing
        for (j, theta) in enumerate(angle_array)
            lines!(ax, h_vals, new_IPRs[j, :]; linestyle=:dash,
                   label="Ry(θ=$(round(theta; digits=3)))")
        end
    end
    axislegend(ax; position=:rc)
    save(joinpath(folder, "ipr_vs_h.png"), fig)

    println("Wrote $folder")
    return folder
end

# ---------------------------------------------------------------------------
# Parameters reproducing the committed raw data.
# ---------------------------------------------------------------------------

const H_COARSE = collect(0.0:0.2:4.0)   # 21 points; every size except 2D 8x8 and 10x10
const H_FINE = collect(0.0:0.1:4.0)     # 41 points; 2D 8x8 and 10x10
const ANGLES = collect(range(0, pi / 2; length=5))

const SIZES_1D = collect(3:100)
const SIZES_2D = collect(3:10)

# 2D 8x8 and 10x10 were run on the fine h grid without the Ry angle sweep.
h_grid_for(geometry::String, L::Int) =
    (geometry == "2d" && L in (8, 10)) ? H_FINE : H_COARSE
angles_for(geometry::String, L::Int) =
    (geometry == "2d" && L in (8, 10)) ? nothing : ANGLES

function parse_args(argv::Vector{String})
    opts = Dict{String,Any}("geometry" => "both", "sizes" => nothing,
                            "seed" => 1234, "outdir" => nothing)
    i = 1
    while i <= length(argv)
        key = argv[i]
        if key in ("--geometry", "--sizes", "--seed", "--outdir")
            i + 1 <= length(argv) || error("$key requires a value")
            opts[key[3:end]] = argv[i + 1]
            i += 2
        else
            error("unknown argument: $key")
        end
    end
    opts["geometry"] = lowercase(String(opts["geometry"]))
    opts["geometry"] in ("1d", "2d", "both") || error("--geometry must be 1d, 2d or both")
    opts["seed"] = opts["seed"] isa String ? parse(Int, opts["seed"]) : opts["seed"]
    if opts["sizes"] !== nothing
        opts["sizes"] = [parse(Int, s) for s in split(String(opts["sizes"]), ',')]
    end
    return opts
end

function main(argv::Vector{String}=String[])
    opts = parse_args(argv)
    Random.seed!(opts["seed"])

    repo_root = dirname(@__DIR__)
    default_root = joinpath(repo_root, "data", "raw")
    root = opts["outdir"] === nothing ? default_root : String(opts["outdir"])

    parameters = Dict{String,Any}(
        "J" => 1.0,
        "boundary_conditions" => "PBC",
        "maxdim" => 400,
        "nsweeps" => 10,
        "cutoff" => 1e-10,
        "num_samples" => 10_000,
    )

    geometries = opts["geometry"] == "both" ? ["1d", "2d"] : [opts["geometry"]]
    for geometry in geometries
        default_sizes = geometry == "1d" ? SIZES_1D : SIZES_2D
        sizes = opts["sizes"] === nothing ? default_sizes : opts["sizes"]
        subdir = joinpath(root, geometry == "1d" ? "tfim_1d" : "tfim_2d")

        for L in sizes
            Lx, Ly = geometry == "1d" ? (L, 1) : (L, L)
            println("\n", "="^70)
            println("TFIM $(uppercase(geometry)): L = $(Lx)x$(Ly)")
            println("="^70)
            sweep_size(Lx, Ly, parameters, h_grid_for(geometry, L),
                       angles_for(geometry, L), subdir)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
