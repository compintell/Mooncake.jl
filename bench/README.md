# Benchmarking

There are two flavours of benchmarks implemented in `run_benchmarks.jl`.
One is a set of pass / fail tests designed to check that large performance regressions are avoided in `Tapir.jl`.
The other is a set of comparisons between a variety of frameworks -- this set is designed to give a rough sense of where `Tapir.jl` stands in comparison to other AD frameworks, and the results should not be thought of as pass / fail tests.

## Tapir-Only Benchmarking

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
1. `tapir_time`: the time is takes `Tapir.jl` to AD the code
1. `Tapir`: `tapir_time / primal_time`
1. `range`: a named tuple with fields `lb` and `ub` specifying the acceptable range of values for `Tapir.jl`.

From here you can look at whatever properties of the results you are interested in.

Note that the types of all of the columns are very simple, so it is fine to write these results out to a csv.
`CSV.jl` is loaded as part of this script, so if you wish to do so, simply write something like
```julia
CSV.write("file_name.csv", df)
```

Additionally, the convenience function `plot_ratio_histogram!` can be used to produce a histogram of `Tapir.jl` with formatting which is suited to this field. Call it as follows:
```julia
derived_results = benchmark_derived_rrules!!(Xoshiro)
df = DataFrame(df)
plot_ratio_histogram!(df)
```

## Inter-framework Benchmarking

This comprises a small suite of functions that we AD using `Tapir.jl`, `Zygote.jl`, `ReverseDiff.jl`, and `Enzyme.jl`.
The primary purpose of this suite of benchmarks is to ensure that we're regularly comparing the performance of a range of reverse-mode ADs on a set of problems which are known to stretch them in various ways.
For any given function in the suite, some frameworks might have rules for it, and some not.
Consequently, they're not comparing the same thing in all cases.

Please note that we have found that the results of the comparisons vary widely from machine to machine.

This suite of benchmarks is also run as part of CI, and the output is recorded in two ways:
1. a table of results is posted as comment in a PR
1. the table and a corresponding graph are stored as github actions artifacts, and can be retrieved by going to the "Checks" tab of your PR, and clicking on the artifact button.

As with the pass / fail tests, these tests report the ratio of the time taken to perform AD to the time taken to run the function being tested.

If you wish to add to this suite, see the `generate_inter_framework_tests` function in `run_benchmarks.jl`.
To run this suite locally, include `run_benchmarks.jl` and run the `create_inter_ad_benchmarks` function.
This will output the graph and table mentioned above to the `bench` folder.
