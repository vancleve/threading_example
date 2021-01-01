using LinearAlgebra
using StatsBase
using Random
using ProgressMeter

include("structures.jl")

# equilibrium abundance from eqn 3 of O'Dwyer (2020)
function nstar!(pop::Population)
    μ = pop.params.μ
    pop.ns .= 1/μ * (pop.M \ pop.ρ)
end

function update_P_DR!(pop::Population)
    λ = pop.params.λ
    μ = pop.params.μ

    for i in 1:pop.size
        pop.α[:,i] .= pop.members[i].traits
    end

    # direct reciprocity model for P
    @. pop.P = (pop.α + λ * pop.α') / (1 - λ^2)
    pop.P[diagind(pop.P)] .= 0.0
    pop.M .= I - 1/μ .* (pop.P .- Diagonal(vec(sum(pop.P, dims=1))))
end

function mutate!(i::Int, pop::Population)
    if pop.ntraits == 1
        if rand() < pop.params.mutrate
            pop.members[i].traits[1] = rand(pop.params.mutdist(pop.members[i].traits[1]))
        end
    else
        inds = [j != i && rand() < pop.params.mutrate for j in 1:pop.ntraits]
        pop.members[i].traits[inds] .= [rand(pop.params.mutdist(t)) for t in pop.members[i].traits[inds]]
    end
end

function reproduce_moran!(pop::Population)
    # equilibrium "abundance" given environmental state
    nstar!(pop)
    # mortality weights as a function of abundance
    mortalityw = ProbabilityWeights(pop.params.mortality.(pop.ns.-pop.params.a/pop.params.b))
    # death due to mortality
    dead = sample(1:pop.size, mortalityw)

    # sample parent randomly from remaining individuals
    parent = sample(setdiff(1:pop.size, dead))

    # clone offspring from parent into empty site
    if pop.ntraits > 1
        others = setdiff(1:pop.size,[dead,parent])
        pop.members[dead].traits[others] .= pop.members[parent].traits[others]
        pop.members[dead].traits[parent] .= pop.members[parent].traits[dead]
    else
        pop.members[dead].traits .= pop.members[parent].traits
    end

    # mutate offspring
    mutate!(dead, pop)
end

function evolve_moran!(pop::Population)
    # update exchange matrix P
    update_P_DR!(pop)
    # death, birth, and mutation
    reproduce_moran!(pop)
end

function epoch!(pop::Population, rp::RunParams, e::Int, prog::Progress)
    nsteps = rp.nsteps
    rec_steps = rp.rec_steps

    # update environmental state
    pop.ρ .= rand(pop.params.ρdist, pop.size)

    # evolve for nsteps time steps during epoch
    for i in 1:nsteps
        pop.params.evolve_func(pop)

        pop.status.step = (e-1) * nsteps + i
        # save data every rec_steps
        if pop.status.step % rec_steps == 0
            rp.save_data(rp.data, pop)
        end
        next!(prog)
    end
end

function evolve_epochs!(pop::Population, rp::RunParams, prog::Progress)
    for e in 1:rp.nepochs
        epoch!(pop, rp, e, prog)
    end
end

function evolve_replicates!(pop::Population, rp::RunParams)
    pops = [Population(pop.params) for r in 1:rp.nreps]
    rps = [deepcopy(rp) for r in 1:rp.nreps]

    prog = Progress(rp.nreps*rp.nepochs*rp.nsteps, desc="replicate...", barlen=80, color=:color_normal, start=rp.nreps*rp.nepochs*rp.nsteps)
    for r in 1:rp.nreps
        # run replicate
        pops[r].status.rep = r
        evolve_epochs!(pops[r], rps[r], prog)
    end
end