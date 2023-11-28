rand_inputs(rng, f, arity) = randn(rng, arity)
rand_inputs(rng, ::typeof(acosh), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, ::typeof(asech), _) = (rand(rng) * 0.9, )
rand_inputs(rng, ::typeof(log), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, ::typeof(asin), _) = (rand(rng) * 0.9, )
rand_inputs(rng, ::typeof(asecd), _) = (rand(rng) + 1, )
rand_inputs(rng, ::typeof(log2), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, ::typeof(log10), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, ::typeof(acscd), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, ::typeof(log1p), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, ::typeof(acsc), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, ::typeof(atanh), _) = (2 * 0.9 * rand(rng) - 0.9, )
rand_inputs(rng, ::typeof(acoth), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, ::typeof(asind), _) = (0.9 * rand(rng), )
rand_inputs(rng, ::typeof(asec), _) = (rand(rng) + 1.001, )
rand_inputs(rng, ::typeof(acosd), _) = (2 * 0.9 * rand(rng) - 0.9, )
rand_inputs(rng, ::typeof(acos), _) = (2 * 0.9 * rand(rng) - 0.9, )
rand_inputs(rng, ::typeof(sqrt), _) = (rand(rng) + 1e-3, )

@testset "low_level_maths" begin
    @testset "$f" for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
        if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f)) ||
            M == :SpecialFunctions
            continue  # Skip rules for methods not defined in the current scope
        end
        arity > 2 && continue
        (f == :rem2pi || f == :ldexp || f == :(^)) && continue
        f = @eval $M.$f
        rng = sr(123456)
        for _ in 1:10
            test_rrule!!(rng, f, rand_inputs(rng, f, arity)...; perf_flag=:stability)
        end
    end
end    
