@testset "foreigncall" begin
    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array,
        :jl_array_grow_end, :jl_array_del_end, :jl_array_copy, :jl_type_intersection,
        :memset, :jl_get_tls_world_age, :memmove, :jl_object_id,
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

    _a, _da = randn(5), randn(5)
    _b, _db = randn(4), randn(4)
    ptr_a, ptr_da = pointer(_a), pointer(_da)
    ptr_b, ptr_db = pointer(_b), pointer(_db)

    function unsafe_copyto_tester(x::Vector{T}, y::Vector{T}, n::Int) where {T}
        GC.@preserve x y unsafe_copyto!(pointer(x), pointer(y), n)
        return x
    end

    # Test the rules which wrap foreign calls in order to prevent them from ever being hit.
    @testset "$f, $(typeof(x))" for (interface_only, perf_flag, f, x...) in [

        # Rules to avoid foreigncall nodes:
        (false, :stability, Base.allocatedinline, Float64),
        (false, :stability, Base.allocatedinline, Vector{Float64}),
        # (true, :none, pointer_from_objref, _x),
        # (
        #     true,
        #     unsafe_pointer_to_objref,
        #     CoDual(
        #         pointer_from_objref(_x),
        #         bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
        #     ),
        # ),
        (true, :stability, Array{Float64, 1}, undef, 5),
        (true, :stability, Array{Float64, 2}, undef, 5, 4),
        (true, :stability, Array{Float64, 3}, undef, 5, 4, 3),
        (true, :stability, Array{Float64, 4}, undef, 5, 4, 3, 2),
        (true, :stability, Array{Float64, 5}, undef, 5, 4, 3, 2, 1),
        (true, :stability, Array{Float64, 4}, undef, (2, 3, 4, 5)),
        (true, :stability, Array{Float64, 5}, undef, (2, 3, 4, 5, 6)),
        (true, :stability, Base._growend!, randn(5), 3),
        (false, :stability, copy, randn(5, 4)),
        (false, :stability, typeintersect, Float64, Int),
        (false, :stability, fill!, rand(Int8, 5), Int8(2)),
        (false, :stability, fill!, rand(UInt8, 5), UInt8(2)),
        (false, :none, Core.Compiler.return_type, sin, Tuple{Float64}),
        (false, :none, Core.Compiler.return_type, Tuple{typeof(sin), Float64}),
        (true, :stability, unsafe_copyto!, CoDual(ptr_a, ptr_da), CoDual(ptr_b, ptr_db), 4),
        (false, :stability, unsafe_copyto!, randn(4), 2, randn(3), 1, 2),
        (false, :stability, unsafe_copyto!, [rand(3) for _ in 1:5], 2, [rand(4) for _ in 1:4], 1, 3),
        (false, :stability, objectid, 5.0),
        (true, :stability, objectid, randn(5)),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [
        (false, reshape, randn(5, 4), (4, 5)),
        (false, reshape, randn(5, 4), (2, 10)),
        (false, reshape, randn(5, 4), (10, 2)),
        (false, reshape, randn(5, 4), (5, 4, 1)),
        (false, reshape, randn(5, 4), (2, 10, 1)),
        (false, unsafe_copyto_tester, randn(5), randn(3), 2),
        (false, unsafe_copyto_tester, randn(5), randn(6), 4),
        (false, unsafe_copyto_tester, [randn(3) for _ in 1:5], [randn(4) for _ in 1:6], 4),
    ]
        rng = Xoshiro(123456)
        test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only, perf_flag=:none)
    end
end
