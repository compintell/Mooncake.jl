@testset "foreigncall" begin
    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array,
        :jl_array_grow_end, :jl_array_del_end, :jl_array_copy, :jl_type_intersection,
        :memset,
    ]
        @test_throws(
            ErrorException,
            Taped.rrule!!(
                CoDual(Umlaut.__foreigncall__, NoTangent()),
                CoDual(Val(name), NoTangent()),
            )
        )
    end

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [

        # Rules to avoid foreigncall nodes:
        (false, Base.allocatedinline, Float64),
        (false, Base.allocatedinline, Vector{Float64}),
        # (true, pointer_from_objref, _x),
        # (
        #     true,
        #     unsafe_pointer_to_objref,
        #     CoDual(
        #         pointer_from_objref(_x),
        #         bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
        #     ),
        # ),
        (true, Array{Float64, 1}, undef, 5),
        (true, Array{Float64, 2}, undef, 5, 4),
        (true, Array{Float64, 3}, undef, 5, 4, 3),
        (true, Array{Float64, 4}, undef, 5, 4, 3, 2),
        (true, Array{Float64, 5}, undef, 5, 4, 3, 2, 1),
        (true, Array{Float64, 4}, undef, (2, 3, 4, 5)),
        (true, Array{Float64, 5}, undef, (2, 3, 4, 5, 6)),
        (true, Base._growend!, randn(5), 3),
        (false, copy, randn(5, 4)),
        (false, typeintersect, Float64, Int),
        (false, fill!, rand(Int8, 5), Int8(2)),
        (false, fill!, rand(UInt8, 5), UInt8(2)),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only)
    end
end