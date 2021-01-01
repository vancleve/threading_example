using Revise
includet("popfunctions.jl")

using ForwardDiff
using Roots
using StatsPlots
using BenchmarkTools
BLAS.set_num_threads(1)

## Test threading

Random.seed!(314)

n = 25

pop = Population(Params(size=n, λ=0.8, β=100.0, a=1, b=1, 
                        mutrate=0.01, mutwidth=1e-2, 
                        evolve_func=evolve_moran!, 
                        init_traits=0.02*ones(1,1)))

rp = RunParams(
    nreps=24,
    nepochs=10000,
    nsteps=1,
    rec_steps=100,
    data=(),
    save_data=(data,pop)->()
)

@time evolve_replicates!(pop, rp)

##