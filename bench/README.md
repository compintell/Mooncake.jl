# Benchmarking

The benchmarking runs as part of CI, and evaluates a sequence of pass / fail tests.

These pass fail tests are useful for catching substantial performance regressions before
they get merged, but necessarily hide a lot of detail.

If you wish to investigate the results yourself locally, you should start a Julia session,
activate this directory, and run:
```julia
include("bench/run_benchmarks.jl")
df = DataFrame(benchmark_derived_rrules!!(Xoshiro))
```
to run the benchmarks which test the performance of AD. This will produce a `DataFrame`
containing a run-down of the results. It has the following columns:
1. `tag`: a `String` with an automatically generated name for the test
1. `primal_time`: the time taken to run the original code
1. `value_and_pb_time`: the time taken to run the forwards and reverse passes
1. `value_and_pb_ratio`: `value_and_pb_time / primal_time`
1. `range`: a named tuple with fields `lb` and `ub` specifying the acceptable range of values for `value_and_pb_ratio`.

From here you can look at whatever properties of the results you are interested in.

Note that the types of all of the columns are very simple, so it is fine to write these results out to a csv.
`CSV.jl` is loaded as part of this script, so if you wish to do so, simply write something like
```julia
CSV.write("file_name.csv", df)
```

Additionally, the convenience function `plot_ratio_histogram!` can be used to produce a histogram of `value_and_pb_ratio` with formatting which is suited to this field. Call it as follows:
```julia
derived_results = benchmark_derived_rrules!!(Xoshiro)
df = DataFrame(df)
plot_ratio_histogram!(df)
```
