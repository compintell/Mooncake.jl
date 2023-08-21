@testset "misc" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [

        # Legit performance rules:
        (false, sin, 5.0),
        (false, cos, 5.0),

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

        # Lack of activity-analysis rules:
        (false, Base.elsize, randn(5, 4)),
        (false, Base.elsize, view(randn(5, 4), 1:2, 1:2)),
        (false, Core.Compiler.sizeof_nothrow, Float64),
        (false, Base.datatype_haspadding, Float64),

        # Performance-rules that would ideally be completely removed.
        (false, size, randn(5, 4)),
        (false, LinearAlgebra.lapack_size, 'N', randn(5, 4)),
        (false, Base.require_one_based_indexing, randn(2, 3), randn(2, 1)),
        (false, in, 5.0, randn(4)),
        (false, iszero, 5.0),
        (false, isempty, randn(5)),
        (false, isbitstype, Float64),
        (false, sizeof, Float64),
        (false, promote_type, Float64, Float64),
    ]
        test_rrule!!(
            Xoshiro(123456), f, x...;
            interface_only, check_conditional_type_stability=false,
        )
    end
end
