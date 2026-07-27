# Consistency checks for data_generation/generate_data_tfim.jl
#
# The reference is an independent dense Kronecker-product construction of
#
#     H = -J sum_<ij> Z_i Z_j - h sum_i X_i
#
# on the same periodic lattice, diagonalized exactly. That pins the bond structure
# (in particular the wrap-around bonds, which are what distinguishes a torus from a
# cylinder), the field term, and every observable the generator writes.
#
# IPRs are Monte Carlo estimates, so they are compared against the exact value using
# the exact standard error of the estimator rather than a fixed tolerance: perfect
# sampling draws x from p(x) = |psi(x)|^2 and averages p(x), giving
# Var = sum_x p^3 - IPR^2.
#
# Usage:
#   julia --project=. tests/data_generation/test_generate_data_tfim.jl

using Test
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "data_generation", "generate_data_tfim.jl"))

const I2 = [1.0 0.0; 0.0 1.0]
const SX = [0.0 1.0; 1.0 0.0]
const SZ = [1.0 0.0; 0.0 -1.0]
const HAD = [1.0 1.0; 1.0 -1.0] / sqrt(2.0)

"""
    kron_op(N, ops)

Dense operator acting as `ops[i]` on site `i` and as the identity elsewhere. Site 1
is the most significant factor, matching the ITensors site ordering.
"""
function kron_op(N::Int, ops::Dict{Int,Matrix{Float64}})
    out = ones(1, 1)
    for i in 1:N
        out = kron(out, get(ops, i, I2))
    end
    return out
end

"""
    reference_bonds(Lx, Ly)

Site-index pairs of the periodic lattice, derived independently of the generator
except for reusing `snake_index` to name the sites. A wrap bond is added along any
direction of extent above two.
"""
function reference_bonds(Lx::Int, Ly::Int)
    bonds = Tuple{Int,Int}[]
    for y in 1:Ly, x in 1:Lx
        i = snake_index(x, y, Lx, Ly)
        if x < Lx
            push!(bonds, (i, snake_index(x + 1, y, Lx, Ly)))
        elseif Lx > 2
            push!(bonds, (i, snake_index(1, y, Lx, Ly)))
        end
        if y < Ly
            push!(bonds, (i, snake_index(x, y + 1, Lx, Ly)))
        elseif Ly > 2
            push!(bonds, (i, snake_index(x, 1, Lx, Ly)))
        end
    end
    return bonds
end

function reference_hamiltonian(Lx::Int, Ly::Int, h::Float64; J::Float64=1.0)
    N = Lx * Ly
    H = zeros(2^N, 2^N)
    for (i, j) in reference_bonds(Lx, Ly)
        H .-= J .* kron_op(N, Dict(i => SZ, j => SZ))
    end
    for i in 1:N
        H .-= h .* kron_op(N, Dict(i => SX))
    end
    return H
end

"""
    ipr_and_stderr(psi, num_samples)

Exact IPR of a dense state vector and the standard error of the perfect-sampling
estimator the generator uses.
"""
function ipr_and_stderr(psi::Vector{Float64}, num_samples::Int)
    p = abs2.(psi)
    p ./= sum(p)
    ipr = sum(p .^ 2)
    variance = sum(p .^ 3) - ipr^2
    return ipr, sqrt(max(variance, 0.0) / num_samples)
end

rotate_to_x(psi::Vector{Float64}, N::Int) = kron(fill(HAD, N)...) * psi

const NUM_SAMPLES = 4000
const PARAMS = Dict{String,Any}(
    "J" => 1.0, "boundary_conditions" => "PBC", "maxdim" => 400,
    "nsweeps" => 12, "cutoff" => 1e-12, "num_samples" => NUM_SAMPLES,
)

@testset "generate_data_tfim.jl" begin

    @testset "single-site operators" begin
        sites = siteinds("S=1/2", 1)
        s = sites[1]
        # Array with an explicit index order, since an ITensor carries no row/column
        # convention of its own.
        as_matrix(t) = Array(t, prime(s), s)

        @test as_matrix(op("H", s)) ≈ HAD
        @test as_matrix(op("Ry", s; θ=0.0)) ≈ I2
        # Ry(pi/2) rotates the Z eigenbasis onto the X eigenbasis, so it agrees with
        # the Hadamard up to the sign of one column.
        ry = as_matrix(op("Ry", s; θ=pi / 2))
        @test abs.(ry) ≈ abs.(HAD)
        @test ry * transpose(ry) ≈ I2
    end

    @testset "snake ordering is a bijection" begin
        for (Lx, Ly) in [(4, 1), (3, 3), (5, 4)]
            idx = [snake_index(x, y, Lx, Ly) for y in 1:Ly for x in 1:Lx]
            @test sort(idx) == collect(1:(Lx * Ly))
        end
    end

    @testset "bond count is periodic, not open" begin
        # A torus has one bond per site per direction of extent above two.
        @test length(reference_bonds(6, 1)) == 6      # ring
        @test length(reference_bonds(4, 4)) == 32     # 2 * 16
        @test length(reference_bonds(3, 3)) == 18
    end

    @testset "ground state matches exact diagonalization" begin
        Random.seed!(20240)
        for (Lx, Ly, h) in [(8, 1, 2.0), (10, 1, 3.0), (3, 3, 3.0)]
            N = Lx * Ly
            eigen_ref = eigen(Symmetric(reference_hamiltonian(Lx, Ly, Float64(h))))
            energy_ref = eigen_ref.values[1]
            gap = eigen_ref.values[2] - eigen_ref.values[1]
            psi_ref = Vector{Float64}(eigen_ref.vectors[:, 1])
            @test gap > 1e-6   # these points are chosen to have a unique ground state

            params = copy(PARAMS)
            params["Lx"] = Lx; params["Ly"] = Ly; params["h"] = Float64(h)
            energy, psi, H, sites = run_dmrg(params; verbose=false)
            orthogonalize!(psi, 1); normalize!(psi)

            @test isapprox(energy, energy_ref; atol=1e-8)

            mx = real(inner(psi', make_M_MPO(sites; direction="X"), psi))
            mx_ref = sum(real(psi_ref' * kron_op(N, Dict(i => SX)) * psi_ref) for i in 1:N)
            @test isapprox(mx, mx_ref; atol=1e-6)

            # IPR in both bases, judged against the exact sampling standard error.
            ipr_z_ref, se_z = ipr_and_stderr(psi_ref, NUM_SAMPLES)
            @test abs(compute_IPR(psi; num_samples=NUM_SAMPLES) - ipr_z_ref) < 6 * se_z

            psi_x = apply(MPO(sites, "H"), psi)
            orthogonalize!(psi_x, 1); normalize!(psi_x)
            ipr_x_ref, se_x = ipr_and_stderr(rotate_to_x(psi_ref, N), NUM_SAMPLES)
            @test abs(compute_IPR(psi_x; num_samples=NUM_SAMPLES) - ipr_x_ref) < 6 * se_x
        end
    end

    @testset "Ry sweep reproduces the analytic product state at h = 0" begin
        # At zero field DMRG returns a polarized product state, whose IPR after
        # Ry(theta) on every site is (cos^4(theta/2) + sin^4(theta/2))^N exactly.
        Random.seed!(7)
        Lx = 8
        params = copy(PARAMS)
        params["Lx"] = Lx; params["Ly"] = 1; params["h"] = 0.0
        _, psi, _, sites = run_dmrg(params; verbose=false)
        orthogonalize!(psi, 1); normalize!(psi)
        @test isapprox(compute_IPR(psi; num_samples=NUM_SAMPLES), 1.0; atol=1e-12)

        for theta in ANGLES
            analytic = (cos(theta / 2)^4 + sin(theta / 2)^4)^Lx
            measured = compute_IPR(apply_ry(psi, sites, theta); num_samples=NUM_SAMPLES)
            # A product state has zero sampling variance in this estimator only at
            # theta = 0; elsewhere allow the exact standard error.
            p = [cos(theta / 2)^2, sin(theta / 2)^2]
            probs = [prod(getindex.(Ref(p), digits(k; base=2, pad=Lx) .+ 1))
                     for k in 0:(2^Lx - 1)]
            se = sqrt(max(sum(probs .^ 3) - analytic^2, 0.0) / NUM_SAMPLES)
            @test abs(measured - analytic) < 6 * se + 1e-12
        end
    end

    @testset "grid selection matches the committed folders" begin
        raw = joinpath(@__DIR__, "..", "..", "Data", "Raw", "Data TFIM")
        for (geometry, subdir, sizes) in [("1d", "Data TFIM 1D", SIZES_1D),
                                          ("2d", "Data TFIM 2D", SIZES_2D)]
            for L in sizes
                folder = joinpath(raw, subdir,
                                  geometry == "1d" ? "L=$(L)x1" : "L=$(L)x$(L)")
                isdir(folder) || continue
                n_h = countlines(joinpath(folder, "hs.csv"))
                @test length(h_grid_for(geometry, L)) == n_h
                has_angles = isfile(joinpath(folder, "new_iprs.csv"))
                @test (angles_for(geometry, L) !== nothing) == has_angles
            end
        end
    end
end
