@testset "avoiding_non_differentiable_code" begin

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [
        # Rules to avoid pointer type conversions.
        (
            true,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),    
            2,
        ),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only)
    end
end
