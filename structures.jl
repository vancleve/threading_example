using Parameters
using Distributions
using SpecialFunctions
import StatsFuns: normcdf, norminvcdf
import Random: AbstractRNG
using DataFrames

struct CDFNormal <: Sampleable{Univariate,Continuous}
    u::Float64
    σ::Float64
    ndist::Distribution

    CDFNormal(u::Float64, σ::Float64) = new(u, σ, Normal(norminvcdf(u), σ))
end

function Base.rand(rng::AbstractRNG, s::CDFNormal)
    return normcdf(rand(rng, s.ndist))
end

# population parameters data type
@with_kw struct Params
    size::Int = 20
    init_traits::Array{Float64} = zeros(size)
    μ::Float64 = 1.0
    a::Float64 = 1.0
    b::Float64 = 1.0
    λ::Float64 = 0.8
    β::Float64 = 100.0
    η::Float64 = 0.001
    ρdist::Distribution = Gamma(a,1/b)
    mutrate::Float64 = 0.01
    mutwidth::Float64 = 0.0001
    mutdist::Function = (u)->CDFNormal(u, mutwidth)
    mortality::Function = (x)->exp(-β*x)
    ntrait_samples::Int = size
    evolve_func::Function
end

@with_kw struct RunParams
    nsteps::Int = 1
    nepochs::Int = 500
    nreps::Int = 1
    rec_steps::Int = 1
    data = ()
    save_data::Function = (data,pop)->()
end

# individual data type
struct Individual
    traits::Array{Float64,1}
end

# struct to hold time step and replicate values for a population
mutable struct Status
    step::Int
    rep::Int
end

# population data type
struct Population
    size::Int
    ntraits::Int
    members::Vector{Individual}
    ρ::Array{Float64,1}
    ns::Array{Float64,1}
    P::Array{Float64,2}
    M::Array{Float64,2}
    A::Array{Float64,2}
    α::Array{Float64,2}
    params::Params
    status::Status

    function Population(params::Params)
        if ndims(params.init_traits) == 1
            ntraits = length(params.init_traits)
            nvals = 1
        else
            ntraits, nvals = size(params.init_traits)
        end
        @assert (ntraits == params.size || ntraits == 1) "num traits should be 1 or pop size elements"
        @assert (nvals == params.size || nvals == 1) "num columns should be 1 or pop size elements"

        members = Vector{Individual}(undef,params.size)
        for i in 1:params.size
            # members[i] = Individual(copy(params.init_traits), 0.0)
            if nvals == 1
                members[i] = Individual(params.init_traits[:])
            else
                members[i] = Individual(params.init_traits[:,i])
            end
        end
        # fitness = [members[i].fitness for i in 1:size]
        ρ = zeros(params.size)
        ns = zeros(params.size)
        P = zeros(params.size, params.size)
        M = zeros(params.size, params.size)
        A = zeros(params.size, params.size)
        α = zeros(params.size, params.size)

        # return new(size, members, fitness, ρ, ns, P, M, A, params, Ref(0))
        return new(params.size, ntraits, members, ρ, ns, P, M, A, α, params, Status(0, 0))
    end
end
