@testset "foreigncall" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:foreigncall))

    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array,
        :jl_array_copy, :jl_type_intersection, :memset, :jl_get_tls_world_age, :memmove,
        :jl_object_id, :jl_array_sizehint, :jl_array_grow_beg, :jl_array_grow_end,
        :jl_array_grow_at, :jl_array_del_beg, :jl_array_del_end, :jl_array_del_at,
        :jl_value_ptr,
    ]
        @test_throws(
            ErrorException,
            Taped.rrule!!(zero_codual(Umlaut.__foreigncall__), zero_codual(Val(name))),
        )
        @test_throws(
            ErrorException,
            Taped.rrule!!(zero_codual(Taped._foreigncall_), zero_codual(Val(name))),
        )
    end
end
