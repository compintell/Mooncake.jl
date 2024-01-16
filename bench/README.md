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
2. `primal_time`: the time taken to run the original code
3. `forwards_time`: the time taken to run the forwards-pass
4. `pullback_time`: the time taken to run the pullback
5. `forwards_ratio`: `forwards_time / primal_time`
6. `pullback_ratio`: `pullback_time / primal_time`
7. `forwards_lb` / `forwards_ub` / `pullback_lb` / `pullback_ub`: required lower / upper bounds on the above ratios

From here you can look at whatever properties of the results you are interested in.

Note that the types of all of the columns are very simple, so it is fine to write these results out to a csv.
`CSV.jl` is loaded as part of this script, so if you wish to do so, simply write something like
```julia
CSV.write("file_name.csv", df)
```
