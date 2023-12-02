using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "."))

using BenchmarkTools, DataFrames, Random, ReverseDiff, Taped

using Taped: CoDual, rrule!!, trace, RMC

relu(x) = max(x, zero(x))
mlp(x, W1, W2) = W2 * relu.(W1 * x)
se_loss(x, y, W1, W2) = sum(abs2, y .- mlp(x, W1, W2))

function main()
    rng = Xoshiro(123456)
    N = 5
    Din = 3
    Dh = 5
    Dout = 1
    x = randn(rng, Din, N)
    W1 = randn(rng, Dh, Din)
    W2 = randn(rng, Dout, Dh)
    y = randn(rng, Dout, N)

    args = (x, y, W1, W2)

    rd_dargs = ReverseDiff.gradient(se_loss, args)
    y, dargs = Taped.value_and_gradient(se_loss, args...)

    # println("Primal timing")
    # display(@benchmark se_loss($args...))
    # println()

    # println("RD No Compilation")
    # display(@benchmark ReverseDiff.gradient(se_loss, $args))
    # println()

    # # Compiled version of ReverseDiff tape.
    # rd_tape = ReverseDiff.compile(ReverseDiff.GradientTape(se_loss, args))
    # println("RD Compiled Tape")
    # display(@benchmark ReverseDiff.gradient!($rd_tape, $args))
    # println()

    # println("Taped timing")
    # display(@benchmark Taped.value_and_gradient(se_loss, $args...))
    # println()

    # tape = last(trace(se_loss, args...; ctx=RMC()))
    # println("Taped Part-Compiled")
    # display(@benchmark Taped.value_and_gradient($tape, se_loss, $args...))
    # println()

    y, tape = Taped.trace(se_loss, args...; ctx=Taped.RMC())
    f_ur = Taped.UnrolledFunction(tape)
    x = (se_loss, args...)
    dx = map(zero_tangent, x)
    x_dx = map(CoDual, x, dx)
    fast_tape = Taped.construct_accel_tape(CoDual(f_ur, NoTangent()), x_dx...)
    ȳ = randn_tangent(rng, y)
    @show length(fast_tape.tape.instructions)
    println("Taped Compiled")
    display(@benchmark Taped.execute!($fast_tape, $ȳ, $x_dx...))
    println()

    @show length(tape) # 6462
    @show count(op -> op isa Umlaut.Constant, tape.ops) # 2001
    @show count(op -> op isa Umlaut.Call && op.fn == Core.apply_type, tape.ops) # 178
    @show count(op -> op isa Umlaut.Call && op.fn == Umlaut.check_variable_length, tape.ops) # 708
    @show count(op -> op isa Umlaut.Call && op.fn == Taped.verify, tape.ops) # 259

    calls = tape.ops[findall(Base.Fix2(isa, Umlaut.Call), tape.ops)]
    call_fns = map(Base.Fix2(getproperty, :fn), calls)

    unique_call_fns = unique(call_fns)
    counts = [count(==(element), call_fns) for element in unique_call_fns]
    df = DataFrame(:counts => counts, :call_fns => unique_call_fns)
    sort!(df, :counts; rev=true)

    # Profiling suggests that about 90% of my execution time is spent in `increment_field!!`,
    # presumably due to the substantial type-instability in it.
    # The plan to improve this is to first make `increment_field!!` a`Val{:f}`
    @profview [Taped.execute!(fast_tape, ȳ, x_dx...) for _ in 1:100]
end

function simple_bench()
    rng = Xoshiro(123456)
    _, f, args... = Taped.TestResources.TEST_FUNCTIONS[2]
    y, tape = Taped.trace(f, args...; ctx=Taped.RMC())
    display(tape)
    println()
    f_ur = Taped.UnrolledFunction(tape)
    x = (f, args...)
    dx = map(zero_tangent, x)
    x_dx = map(CoDual, x, dx)
    fast_tape, rev_tape, new_tape = Taped.construct_accel_tape(CoDual(f_ur, NoTangent()), x_dx...);
    ȳ = randn_tangent(rng, y)
    @show length(fast_tape.tape.instructions)
    println("Taped Pointer Compiled")
    display(@benchmark Taped.execute!($fast_tape, $ȳ, $x_dx...))
    println()

    println("Taped Julia Compiled")
    display(@benchmark Taped.execute!($rev_tape, $ȳ, $x_dx...))
    println()

    # Profiling suggests that about 90% of my execution time is spent in `increment_field!!`,
    # presumably due to the substantial type-instability in it.
    # The plan to improve this is to first make `increment_field!!` a`Val{:f}`
    Taped.execute!(fast_tape, ȳ, x_dx...)
    [Taped.execute!(fast_tape, ȳ, x_dx...) for _ in 1:100_000]
    @profview [Taped.execute!(fast_tape, ȳ, x_dx...) for _ in 1:1_000_000]

    # Benchmark each instruction individually to determine where an allocation is happening.
    raw_tape = fast_tape.tape.instructions;
    results = map(enumerate(raw_tape)) do (n, inst)
        @show n
        @benchmark $inst() samples=1 evals=1
    end
    idx = findall(x -> allocs(x) > 0, results)
    @show results[idx]
    @show idx

    Taped.execute!(rev_tape, ȳ, x_dx...)
end

# Tape{Taped.ReverseModeADContext}
#   inp %1::Taped.CoInstruction{Tuple{}, Base.RefValue{CoDual{typeof(Taped.TestResources.test_while_loop), Tangent{NamedTuple{(), Tuple{}}}}}, Nothing}
#   inp %2::Taped.CoInstruction{Tuple{}, Base.RefValue{CoDual{Float64, Float64}}, Nothing}
#   const %3 = 3::Int64
#   %4 = >(%3, 0)::Bool 
#   %5 = verify(Taped.ConditionalCheck{Bool}(true), %4)::Nothing 
#   %6 = rebind(%2)::Float64 
#   %7 = +(%2, %6)::Float64 
#   %8 = -(%3, 1)::Int64 
#   %9 = >(%8, 0)::Bool 
#   %10 = verify(Taped.ConditionalCheck{Bool}(true), %9)::Nothing 
#   %11 = rebind(%7)::Float64 
#   %12 = +(%7, %11)::Float64 
#   %13 = -(%8, 1)::Int64 
#   %14 = >(%13, 0)::Bool 
#   %15 = verify(Taped.ConditionalCheck{Bool}(true), %14)::Nothing 
#   %16 = rebind(%12)::Float64 
#   %17 = +(%12, %16)::Float64 
#   %18 = -(%13, 1)::Int64 
#   %19 = >(%18, 0)::Bool 
#   %20 = verify(Taped.ConditionalCheck{Bool}(false), %19)::Nothing 

# Before optimising getfield:
# julia> main()
# Taped Compiled
# BenchmarkTools.Trial: 80 samples with 1 evaluation.
#  Range (min … max):  58.504 ms … 93.104 ms  ┊ GC (min … max): 0.00% … 22.92%
#  Time  (median):     60.936 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   63.007 ms ±  6.497 ms  ┊ GC (mean ± σ):  1.28% ±  4.47%

#     ▅█    ▁                                                    
#   ▆▆██▇▆▆▆█▆▁▄▁▃▃▁▃▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▃ ▁
#   58.5 ms         Histogram: frequency by time        92.5 ms <

#  Memory estimate: 2.72 MiB, allocs estimate: 83643.

# After optimising getfield:
# Taped Compiled
# BenchmarkTools.Trial: 162 samples with 1 evaluation.
#  Range (min … max):  21.028 ms … 88.329 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     26.483 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   30.978 ms ± 11.160 ms  ┊ GC (mean ± σ):  2.66% ± 8.37%

#   ▆ ▁▁█▅██▁                                                    
#   █████████▇▆▅▁▆▃▃▃▄▄▄▆▃▁▃▁▁▅▄▄▄▄▄▃▁▄▃▁▄▁▃▄▁▃▁▁▁▃▁▁▃▃▁▄▁▁▃▁▁▃ ▃
#   21 ms           Histogram: frequency by time        64.1 ms <

#  Memory estimate: 2.21 MiB, allocs estimate: 69098.

# Optimising getfield appears to have been a resounding success.
# The next bottleneck operation _appears_ to be `might_be_active`.
# I'm not currently completely sure what is preventing it from being
# performant -- so I need to figure out how to debug.
# In any case, it is entirely a function of the types of the arguments, so it
# really ought to work...
#
# After forcing Julia to specialise on argument types of optimise_rrule!!:
# julia> main();
# Taped Compiled
# BenchmarkTools.Trial: 348 samples with 1 evaluation.
#  Range (min … max):  11.939 ms … 40.662 ms  ┊ GC (min … max): 0.00% … 61.96%
#  Time  (median):     13.631 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   14.373 ms ±  4.089 ms  ┊ GC (mean ± σ):  4.37% ± 10.03%

#   ▅████▇▅▄                                                     
#   ████████▁▄█▆▄▆▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄▄▆▄ ▇
#   11.9 ms      Histogram: log(frequency) by time      38.9 ms <

#  Memory estimate: 1.95 MiB, allocs estimate: 53145.
#
# So we're doing a lot better, but we're not there yet. Looking at the flame chart, things
# have improved substantially, but it looks like there's still a number of type-
# instabilities floating around in the general code for doing stuff. It really ought to be
# possible to remove this.
#
# By making `might_be_active` a generated function, the performance appears to be _dramatically_
# improved:
# Taped Compiled
# BenchmarkTools.Trial: 2637 samples with 1 evaluation.
#  Range (min … max):  1.436 ms … 24.449 ms  ┊ GC (min … max): 0.00% … 90.12%
#  Time  (median):     1.631 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.887 ms ±  1.784 ms  ┊ GC (mean ± σ):  7.49% ±  7.39%

#    ▂█▄▃▁                                                      
#   ▇█████▇█▇▆▅▅▅▄▆▄▄▅▄▄▃▃▃▃▃▃▃▃▂▂▂▃▂▂▂▃▃▂▂▃▂▂▂▂▂▃▂▂▂▂▂▂▂▂▂▂▁▂ ▃
#   1.44 ms        Histogram: frequency by time        3.07 ms <

#  Memory estimate: 498.47 KiB, allocs estimate: 7634.
#
# It remains unclear whether the accelerated code continues to compute the correct thing
# though, so I need to spend some time on Monday going over what it is doing.
# Note that improving type-stability has not just improved the performance of the
# accelerated tape, it has also substantially improved the performance of all tracing
# operations. Notably, the second time that the tape is run, it now takes about as long
# as the accelerated tape originally took:
# Taped Part-Compiled
# BenchmarkTools.Trial: 54 samples with 1 evaluation.
#  Range (min … max):  87.256 ms … 129.111 ms  ┊ GC (min … max): 0.00% … 22.20%
#  Time  (median):     91.004 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   93.902 ms ±   9.589 ms  ┊ GC (mean ± σ):  2.23% ±  5.95%

#       ▂█▂                                                       
#   ▃█▅█████▆▇▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▃▁▁▃ ▁
#   87.3 ms         Histogram: frequency by time          127 ms <

#  Memory estimate: 5.25 MiB, allocs estimate: 146666.

# By forcing some further specialisation, hence avoiding yet more type instabilities, I've
# managed to get the run time down to:
# length(fast_tape.tape.instructions) = 12926
# Taped Compiled
# BenchmarkTools.Trial: 1568 samples with 1 evaluation.
#  Range (min … max):  2.240 ms … 67.672 ms  ┊ GC (min … max):  0.00% … 94.23%
#  Time  (median):     2.569 ms              ┊ GC (median):     0.00%
#  Time  (mean ± σ):   3.182 ms ±  5.544 ms  ┊ GC (mean ± σ):  15.59% ±  8.53%

#     ▃▆█▆▃▁▁▂ ▁                                                
#   ▃▆████████▇█▆▇▅▆▄▅▅▅▄▄▄▄▄▃▄▃▃▃▃▃▃▃▂▂▂▃▂▂▂▃▂▂▁▂▁▂▂▂▂▁▁▂▁▁▂▂ ▃
#   2.24 ms        Histogram: frequency by time        4.44 ms <

#  Memory estimate: 618.38 KiB, allocs estimate: 9629.
#
# Note that the tape is of length 12926, meaning that we're down to under 1 allocation per
# operation, which suggests to me that there are just some rules / primitives that aren't
# inferring properly, rather than there being some kind of problem with the overall
# mechanism.
#
# I'm having some real issues getting the function wrapper / point things to consistently
# give zero allocations inside code that I really feel ought to have zero allocations.
# Moreover, it appears to be really hard to get my flame graph to do something sensible
# for the compiled functions...
