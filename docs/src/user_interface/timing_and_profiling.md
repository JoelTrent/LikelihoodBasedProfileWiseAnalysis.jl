# Timing and Profiling

Progress meters are implemented using [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl)....


## TimerOutputs.jl

Timing and counting of the number of likelihood function evaluations (and other metrics of interest) are implemented using the `@timeit_debug` macro from [`TimerOutputs.jl`](https://github.com/KristofferC/TimerOutputs.jl). The debug version of the macro means that it is not used within the package (and thus doesn't impact runtime) unless it is enabled using `PlaceholderLikelihood.TimerOutputs.enable_debug_timings(PlaceholderLikelihood)`. Similarly, after being enabled it can be disabled again using `PlaceholderLikelihood.TimerOutputs.disable_debug_timings(PlaceholderLikelihood)`. Enabling debug timings will cause methods that use the macro to recompile.

The macro will only work correctly if distributed computing via [`Distributed.jl`](https://docs.julialang.org/en/v1/stdlib/Distributed/) is not used - the timer on the main worker will not record the timings made on other workers (i.e. `using Distributed; nworkers()` returns `1`). The macro cannot be used for dimensional and full likelihood sampling when their keyword argument `use_threads` is `true` - an exception will be raised.

The exact timings extracted using this macro may not be quite true if function evaluation is very fast due to it's overhead. However, it's main value is in recording the number of function evaluations that are made. 

```
using PlaceholderLikelihood

# model definition ...

PlaceholderLikelihood.TimerOutputs.enable_debug_timings(PlaceholderLikelihood)

PlaceholderLikelihood.TimerOutputs.reset_timer!(PlaceholderLikelihood.timer)

univariate_confidenceintervals!(model)

PlaceholderLikelihood.timer
```