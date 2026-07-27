# Consistency checks for data_generation/generate_data_tfim_thermal_hs_kernel.jl
#
# The reference is exact thermal diagonalization of
#
#     H(h) = -J sum_i Z_i Z_{i+1} - h sum_i X_i,
#     K_beta(h_a, h_b) = Tr[rho_a rho_b] / sqrt(Tr[rho_a^2] Tr[rho_b^2]),
#
# built densely from Kronecker products. The decisive check is the boundary
# condition: this dataset is OPEN, unlike the IPR-kernel dataset, and the tests
# confirm it matches an open reference while disagreeing with a periodic one.
#
# The other convention worth pinning is the normalization: every beta > 0 slice has
# Tr[rho^2] = 1, so the kernel diagonal is exactly one, while the beta = 0 slice is
# the untouched identity MPO and carries Tr[Id^2] = 2^L.
#
# Usage:
#   julia --project=. tests/data_generation/test_generate_data_tfim_thermal_hs_kernel.jl

using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "data_generation",
                 "generate_data_tfim_thermal_hs_kernel.jl"))

const I2 = [1.0 0.0; 0.0 1.0]
const SX = [0.0 1.0; 1.0 0.0]
const SZ = [1.0 0.0; 0.0 -1.0]

function kron_op(N::Int, ops::Dict{Int,Matrix{Float64}})
    out = ones(1, 1)
    for i in 1:N
        out = kron(out, get(ops, i, I2))
    end
    return out
end

function reference_hamiltonian(N::Int, h::Float64; periodic::Bool=false, J::Float64=1.0)
    H = zeros(2^N, 2^N)
    bonds = periodic ? [(i, mod1(i + 1, N)) for i in 1:N] : [(i, i + 1) for i in 1:(N - 1)]
    for (i, j) in bonds
        H .-= J .* kron_op(N, Dict(i => SZ, j => SZ))
    end
    for i in 1:N
        H .-= h .* kron_op(N, Dict(i => SX))
    end
    return H
end

"""
    reference_gibbs(N, h, beta; periodic)

Unnormalized `exp(-beta H)`, with the ground-state energy shifted out for stability.
"""
function reference_gibbs(N::Int, h::Float64, beta::Float64; periodic::Bool=false)
    E = eigen(Symmetric(reference_hamiltonian(N, h; periodic=periodic)))
    return E.vectors * Diagonal(exp.(-beta .* (E.values .- minimum(E.values)))) * E.vectors'
end

"""
    reference_kernel(N, h_a, h_b, beta; periodic)

Hilbert-Schmidt kernel with each state normalized to unit Hilbert-Schmidt norm, which
is the convention the generator produces.
"""
function reference_kernel(N::Int, h_a::Float64, h_b::Float64, beta::Float64;
                          periodic::Bool=false)
    ra = reference_gibbs(N, h_a, beta; periodic=periodic)
    rb = reference_gibbs(N, h_b, beta; periodic=periodic)
    return tr(ra * rb) / sqrt(tr(ra * ra) * tr(rb * rb))
end

"""
    sweep_kernel_block(L, h_vals, beta)

Run the generator's own sweep and read back the block at the final beta.
"""
function sweep_kernel_block(L::Int, h_vals::Vector{Float64}, beta::Float64, dir::String)
    mkpath(dir)
    sweep_size(L, dir; h_vals=h_vals, beta_vals=[0.0, beta])
    rows = readlines(joinpath(dir, "HS Kernel L=$(L).csv"))[2:end]
    n = length(h_vals)
    block = zeros(n, n)
    beta0 = zeros(n, n)
    for row in rows
        f = split(row, ",")
        b = parse(Float64, f[1])
        ia = findfirst(x -> isapprox(x, parse(Float64, f[2]); atol=1e-12), h_vals)
        ib = findfirst(x -> isapprox(x, parse(Float64, f[3]); atol=1e-12), h_vals)
        target = isapprox(b, 0.0; atol=1e-12) ? beta0 : block
        target[ia, ib] = parse(Float64, f[4])
        @test isapprox(parse(Float64, f[5]), 0.0; atol=1e-12)   # kernel is real
    end
    return block, beta0
end

const N_TEST = 8

@testset "generate_data_tfim_thermal_hs_kernel.jl" begin

    @testset "matches open thermal ED, not periodic" begin
        sites = siteinds("S=1/2", N_TEST)
        for beta in (0.5, 2.0)
            states = prepare_ising_gibbs_state(sites, N_TEST, J, 1.0, [Float64(beta)])
            @test length(states) == 1
            # Unit Hilbert-Schmidt norm is the generator's normalization.
            @test isapprox(tr(apply(states[1], states[1]; cutoff=CUTOFF)), 1.0; rtol=1e-8)
        end

        for (h_a, h_b, beta) in [(0.5, 1.0, 0.5), (1.0, 3.0, 0.5), (0.5, 1.0, 2.0)]
            sa = prepare_ising_gibbs_state(sites, N_TEST, J, Float64(h_a), [Float64(beta)])[1]
            sb = prepare_ising_gibbs_state(sites, N_TEST, J, Float64(h_b), [Float64(beta)])[1]
            got = real(tr(apply(sa, sb; maxdim=KERNEL_MAXDIM, cutoff=KERNEL_CUTOFF)))

            open_ref = reference_kernel(N_TEST, Float64(h_a), Float64(h_b), Float64(beta);
                                        periodic=false)
            periodic_ref = reference_kernel(N_TEST, Float64(h_a), Float64(h_b), Float64(beta);
                                            periodic=true)
            # Trotter error at dtau = 0.01 sets the tolerance.
            @test isapprox(got, open_ref; atol=1e-4)
            @test abs(got - periodic_ref) > 1e-3
        end
    end

    @testset "sweep output: diagonal, symmetry and the beta = 0 offset" begin
        dir = mktempdir()
        h_vals = [0.5, 1.0, 2.0]
        block, beta0 = sweep_kernel_block(N_TEST, h_vals, 1.0, dir)

        # Every beta > 0 slice is normalized, so the diagonal is exactly one.
        for i in eachindex(h_vals)
            @test isapprox(block[i, i], 1.0; atol=1e-8)
        end
        @test isapprox(block, block'; atol=1e-8)
        @test all(block .<= 1.0 + 1e-8)

        # The beta = 0 slice is the untouched identity MPO: every entry is 2^L.
        @test all(isapprox.(beta0, 2.0^N_TEST; rtol=1e-10))

        # Against exact ED.
        for (ia, ha) in enumerate(h_vals), (ib, hb) in enumerate(h_vals)
            @test isapprox(block[ia, ib],
                           reference_kernel(N_TEST, ha, hb, 1.0; periodic=false);
                           atol=1e-4)
        end
    end

    @testset "lockstep sweep agrees with per-field construction" begin
        # sweep_size evolves all fields together; prepare_ising_gibbs_state evolves one
        # at a time. They must agree, since that restructuring was done for memory only.
        dir = mktempdir()
        h_vals = [0.5, 1.5]
        block, _ = sweep_kernel_block(N_TEST, h_vals, 0.6, dir)
        sites = siteinds("S=1/2", N_TEST)
        states = [prepare_ising_gibbs_state(sites, N_TEST, J, h, [0.6])[1] for h in h_vals]
        for (ia, _) in enumerate(h_vals), (ib, _) in enumerate(h_vals)
            direct = real(tr(apply(states[ia], states[ib];
                                   maxdim=KERNEL_MAXDIM, cutoff=KERNEL_CUTOFF)))
            @test isapprox(block[ia, ib], direct; atol=1e-10)
        end
    end

    @testset "grid matches the committed files" begin
        path = joinpath(@__DIR__, "..", "..", "Data", "Raw", "Data TFIM Thermal",
                        "Data HS Kernel", "HS Kernel L=10.csv")
        if isfile(path)
            @test countlines(path) - 1 == length(BETA_VALS) * length(H_VALS)^2
            @test split(readline(path), ",") ==
                  ["beta", "h_a", "h_b", "kernel_real", "kernel_imag"]
        end
    end
end
