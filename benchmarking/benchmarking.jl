using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "."))

using BenchmarkTools, Random, ReverseDiff, Taped

using Taped: CoDual, rrule!!, trace, RMC

relu(x) = max(x, zero(x))
mlp(x, W1, W2) = W2 * relu.(W1 * x)
se_loss(x, y, W1, W2) = sum(abs2, y .- mlp(x, W1, W2))

function execute!(tape)
    for inst in tape.instructions
        inst()
    end
    return Taped.get_return_val(tape)
end

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

    println("Primal timing")
    display(@benchmark se_loss($args...))
    println()

    println("RD No Compilation")
    display(@benchmark ReverseDiff.gradient(se_loss, $args))
    println()

    # Compiled version of ReverseDiff tape.
    rd_tape = ReverseDiff.compile(ReverseDiff.GradientTape(se_loss, args))
    println("RD Compiled Tape")
    display(@benchmark ReverseDiff.gradient!($rd_tape, $args))
    println()

    println("Taped timing")
    display(@benchmark Taped.value_and_gradient(se_loss, $args...))
    println()

    tape = last(trace(se_loss, args...; ctx=RMC()))
    println("Taped Part-Compiled")
    display(@benchmark Taped.value_and_gradient($tape, se_loss, $args...))
    println()

    y, tape = Taped.trace(se_loss, args...; ctx=Taped.RMC())
    f_ur = Taped.UnrolledFunction(tape)
    x = (se_loss, args...)
    dx = map(zero_tangent, x)
    x_dx = map(CoDual, x, dx)
    fast_tape = Taped.construct_accel_tape(0.0, CoDual(f_ur, NoTangent()), x_dx...)
    println("Taped Compiled")
    display(@benchmark execute!($fast_tape))
    println()
end
