# Initial Setup

Here we go through a series of examples that show the use of this package on various models.

## Import Package and Set Up Distributed Environment

This package is designed to be run in a distributed computing environment (see [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/)), meaning that the evaluation of profiles can be evaluated in parallel. This is particularly important when running coverage simulations. Many of the methods also work using a multi-threaded approach, albeit often parallised in different locations

!!! tip "Recommended number of workers"
    The number of workers to use is recommended to be one or two less than the number of local threads (typically double the number of cores on a given CPU) on your CPU. This is because 1 worker already exists as the master worker and for most efficient distributed computing we want each worker to work on a distinct CPU thread. Furthermore, if you wish to use your computer at the same time as a simulation is running, we recommend that at least two threads are left available for other tasks.

We first allocate a number of workers for our simulation, which each use the minimum amount of threads (appears to be 2-3 when testing). If we don't set the `JULIA_NUM_THREADS` environment variable to `1` then every worker will be initialised with the default number of threads, `Threads.nthreads()`, which can cause large slowdowns.

```julia
using Distributed
addprocs(Threads.nthreads()-2, env=["JULIA_NUM_THREADS"=>"1"]) # the number of parallel workers to use - on the author's system Threads.nthreads() returns 10
# addprocs(8, env=["JULIA_NUM_THREADS"=>"1"]) # add 8 workers - total number of workers will be 9
```

We can now check how many workers are allocated.

```julia
nprocs() # use to check the total number of worker processes allocated
workers() # the ids of the allocated workers
# Threads.nthreads() # the number of execution threads allocated to Julia at startup
```

We import our package with the annotation `@everywhere`, letting Julia know that we wish to load the package on all allocated workers.

!!! warning "Use of `@everywhere`"
    If `@everywhere` is not used at the beginning of the package import, then the package `PlaceholderLikelihood` is only loaded onto the master worker's environment and cannot be seen by the other workers we wish to parallelise the simulation on. Similarly, any other functions that need to be loaded onto all workers, such as those that define the log-likelihood function, also need to have the annotation `@everywhere`. Resultantly, Julia will throw an error if  `nprocs()>1`. However, an error won't be thrown if we haven't added any worker processes (if `nprocs()==1`) using the above multiprocessing arguments (this package can run on a single thread).

```julia
@everywhere using PlaceholderLikelihood
```