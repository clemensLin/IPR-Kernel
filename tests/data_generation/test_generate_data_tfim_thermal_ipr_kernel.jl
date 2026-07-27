# Consistency checks for data_generation/generate_data_tfim_thermal_ipr_kernel.jl
#
# The reference is exact thermal diagonalization of
#
#     H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
#
# built densely from Kronecker products. The decisive check is the boundary
# condition: this dataset is PERIODIC, and the tests confirm it matches a periodic
# reference while disagreeing with an open one, which is the mistake that would
# otherwise be silent.
#
# The sampler is Monte Carlo, so comparisons use the exact standard error of the
# estimator: it draws x from p(x) = <x|rho|x> and averages p(x), so
# Var = sum_x p^3 - IPR^2.
#
# Usage:
#   julia --project=. tests/data_generation/test_generate_data_tfim_thermal_ipr_kernel.jl

using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "..", "data_generation",
                 "generate_data_tfim_thermal_ipr_kernel.jl"))

const I2 = [1.0 0.0; 0.0 1.0]
const SX = [0.0 1.0; 1.0 0.0]
const SZ = [1.0 0.0; 0.0 -1.0]
const HAD = [1.0 1.0; 1.0 -1.0] / sqrt(2.0)

function kron_op(N::Int, ops::Dict{Int,Matrix{Float64}})
    out = ones(1, 1)
    for i in 1:N
        out = kron(out, get(ops, i, I2))
    end
    return out
end

"""
    reference_hamiltonian(N, h; periodic)

Dense TFIM chain. `periodic` adds the wrap bond, giving `N` bonds instead of `N-1`.
"""
function reference_hamiltonian(N::Int, h::Float64; periodic::Bool=true, J::Float64=1.0)
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
    exact_thermal_iprs(N, h, beta; periodic)

Exact `(ipr_z, ipr_x)` of the Gibbs state, plus the standard errors of the sampling
estimator at `num_samples`, computed from the diagonal of `rho` in each basis.
"""
function exact_thermal_iprs(N::Int, h::Float64, beta::Float64, num_samples::Int;
                            periodic::Bool=true)
    E = eigen(Symmetric(reference_hamiltonian(N, h; periodic=periodic)))
    weights = exp.(-beta .* (E.values .- minimum(E.values)))
    rho = E.vectors * Diagonal(weights) * E.vectors'
    rho ./= tr(rho)

    had = kron(fill(HAD, N)...)
    out = Tuple{Float64,Float64}[]
    for basis_rho in (rho, had * rho * had)
        p = real.(diag(basis_rho))
        p ./= sum(p)
        ipr = sum(p .^ 2)
        se = sqrt(max(sum(p .^ 3) - ipr^2, 0.0) / num_samples)
        push!(out, (ipr, se))
    end
    return out[1], out[2]
end

"""
    thermal_mpo(sites, N, h, beta)

The generator's Gibbs-state MPO at one `(h, beta)`, normalized to unit trace.
"""
function thermal_mpo(sites, N::Int, h::Float64, beta::Float64)
    gates = trotter_gates(sites, N, J, h, DTAU)
    rho = MPO(sites, "Id")
    current = 0.0
    while current < beta - 1e-9
        rho = apply(gates, rho; cutoff=CUTOFF, maxdim=MAXDIM, apply_dag=true)
        current += DTAU
    end
    return rho / tr(rho)
end

const NUM_SAMPLES = 200_000
const N_TEST = 8

@testset "generate_data_tfim_thermal_ipr_kernel.jl" begin

    @testset "infinite temperature is exactly 2^-L" begin
        # At beta = 0 the state is maximally mixed, so IPR = 2^L * (2^-L)^2 = 2^-L
        # independently of the Hamiltonian, in every basis.
        for N in (6, 8)
            sites = siteinds("S=1/2", N)
            rho = MPO(sites, "Id") / tr(MPO(sites, "Id"))
            ipr, _, _ = estimate_ipr_mc(rho, sites; num_samples=2000)
            @test isapprox(ipr, 2.0^(-N); rtol=1e-12)
        end
    end

    @testset "sampler is unbiased on a known diagonal" begin
        # A pure polarized product state has p(x) = delta, so IPR = 1 exactly and the
        # estimator has zero variance. This isolates the sampler from the evolution.
        N = 6
        sites = siteinds("S=1/2", N)
        rho = MPO(sites, "Id")
        proj = [op("ProjUp", sites[j]) for j in 1:N]
        pure = apply(proj, rho; apply_dag=false)
        ipr, _, _ = estimate_ipr_mc(pure, sites; num_samples=500)
        @test isapprox(ipr, 1.0; atol=1e-10)
    end

    @testset "matches periodic thermal ED, not open" begin
        Random.seed!(4242)
        sites = siteinds("S=1/2", N_TEST)
        for (h, beta) in [(0.5, 1.0), (1.0, 2.0), (2.0, 1.5)]
            rho = thermal_mpo(sites, N_TEST, Float64(h), Float64(beta))
            ipr_z, _, _ = estimate_ipr_mc(rho, sites; num_samples=NUM_SAMPLES)
            rho_x = apply([op("H", sites[j]) for j in 1:N_TEST], rho; apply_dag=true)
            ipr_x, _, _ = estimate_ipr_mc(rho_x, sites; num_samples=NUM_SAMPLES)

            (pz, se_z), (px, se_x) = exact_thermal_iprs(N_TEST, Float64(h), Float64(beta),
                                                        NUM_SAMPLES; periodic=true)
            # Trotter error at dtau = 0.01 dominates the sampling error here, so allow
            # a small relative tolerance on top of six sigma.
            @test abs(ipr_z - pz) < 6 * se_z + 1e-3 * pz
            @test abs(ipr_x - px) < 6 * se_x + 1e-3 * px

            # The open-boundary reference must be clearly excluded.
            (oz, _), _ = exact_thermal_iprs(N_TEST, Float64(h), Float64(beta),
                                            NUM_SAMPLES; periodic=false)
            @test abs(ipr_z - oz) > 20 * se_z
        end
    end

    @testset "grid matches the committed files" begin
        path = joinpath(@__DIR__, "..", "..", "Data", "Raw", "Data TFIM Thermal",
                        "Data IPR Kernel", "ipr_thermal_L=10.csv")
        if isfile(path)
            rows = countlines(path) - 1
            @test rows == length(H_VALS) * length(BETA_VALS)
            header = split(readline(path), ",")
            @test header == ["h", "beta", "ipr_z", "ipr_x"]
        end
    end
end
