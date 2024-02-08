# Timing and Profiling

Progress meters are implemented using [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl) and can be used to keep track of how much longer any profile or prediction evaluation has remaining. They can be disabled on a per function basis using the kwarg `show_progress` (which by default is `model.show_progress`). They can be disabled for all functions by setting it to `false` when calling [`initialise_LikelihoodModel`](@ref).

## TimerOutputs.jl

Timing and counting of the number of likelihood function evaluations (and other metrics of interest) are implemented using the `@timeit_debug` macro from [`TimerOutputs.jl`](https://github.com/KristofferC/TimerOutputs.jl). The debug version of the macro means that it is not used within the package (and thus doesn't impact runtime) unless it is enabled using `LikelihoodBasedProfileWiseAnalysis.TimerOutputs.enable_debug_timings(LikelihoodBasedProfileWiseAnalysis)`. Similarly, after being enabled it can be disabled again using `LikelihoodBasedProfileWiseAnalysis.TimerOutputs.disable_debug_timings(LikelihoodBasedProfileWiseAnalysis)`. Enabling debug timings will cause methods that use the macro to recompile.

The macro will only work correctly if distributed computing via [`Distributed.jl`](https://docs.julialang.org/en/v1/stdlib/Distributed/) is not used - the timer on the main worker will not record the timings made on other workers (i.e. `using Distributed; nworkers()` returns `1`). The macro cannot be used for dimensional and full likelihood sampling when their keyword argument `use_threads` is `true` - an exception will be raised.

The exact timings extracted using this macro may not be quite true if function evaluation is very fast due to it's overhead. However, it's main value is in recording the number of function evaluations that are made. An example of this can be seen in [Function Evaluation Timing - Logistic Model](@ref).

```julia
using LikelihoodBasedProfileWiseAnalysis

# model definition ...

LikelihoodBasedProfileWiseAnalysis.TimerOutputs.enable_debug_timings(LikelihoodBasedProfileWiseAnalysis)

LikelihoodBasedProfileWiseAnalysis.TimerOutputs.reset_timer!(LikelihoodBasedProfileWiseAnalysis.timer)

univariate_confidenceintervals!(model)

LikelihoodBasedProfileWiseAnalysis.timer
```