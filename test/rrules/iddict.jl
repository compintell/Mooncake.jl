@testset "iddict" begin
    @testset "IdDict tangent functionality" begin
        p = IdDict(true => 5.0, false => 4.0)
        z = IdDict(true => 3.0, false => 2.0)
        x = IdDict(true => 1.0, false => 1.0)
        y = IdDict(true => 2.0, false => 1.0)
        rng = Xoshiro(123456)
        test_tangent(rng, p, z, x, y)
    end

    @testset "$f, $(typeof(x))" for (interface_only, perf_flag, f, x...) in [
        (false, :stability, Base.rehash!, IdDict(true => 5.0, false => 4.0), 10),
        (false, :none, setindex!, IdDict(true => 5.0, false => 4.0), 3.0, false),
        (false, :none, setindex!, IdDict(true => 5.0), 3.0, false),
        (false, :none, get, IdDict(true => 5.0, false => 4.0), false, 2.0),
        (false, :none, getindex, IdDict(true => 5.0, false => 4.0), true),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
end
