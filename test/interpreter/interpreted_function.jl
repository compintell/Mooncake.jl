@testset "interpreted_function" begin
    @testset "TypedGlobalRef" begin
        @testset "tracks changes" begin
            global __x_for_gref = 5.0
            r = TypedGlobalRef(GlobalRef(Main, :__x_for_gref))
            @test r[] == 5.0
            global __x_for_gref = 4.0
            @test r[] == 4.0
        end
        @testset "is type stable" begin
            global __y_for_gref::Float64 = 5.0
            r = TypedGlobalRef(GlobalRef(Main, :__y_for_gref))
            @test @inferred(r[]) == 5.0
            global __y_for_gref = 4.0
            @test @inferred(r[]) == 4.0
        end
    end

    # Check correctness and performance of the ArgInfo type. We really need everything to
    # infer correctly.
    @testset "ArgInfo: $Tx, $(x), $is_va" for (Tx, x, is_va) in Any[

        # No varargs examples.
        Any[Tuple{Float64}, (5.0,), false],
        Any[Tuple{Float64, Int}, (5.0, 3), false],
        Any[Tuple{Type{Float64}}, (Float64, ), false],
        Any[Tuple{Type{Any}}, (Any, ), false],

        # Varargs examples.
        Any[Tuple{Tuple{Float64}}, (5.0, ), true],
        Any[Tuple{Tuple{Float64, Int}}, (5.0, 3), true],
        Any[Tuple{Float64, Tuple{Int}}, (5.0, 3), true],
        Any[Tuple{Float64, Tuple{Int, Float64}}, (5.0, 3, 4.0), true],
    ]
        ai = Taped.ArgInfo(Tx, is_va)
        @test @inferred Taped.load_args!(ai, x) === nothing
    end

    @testset "TypedPhiNode" begin
        @testset "standard example of a phi node" begin
            node = TypedPhiNode(
                SlotRef{Float64}(),
                SlotRef{Float64}(),
                (1, 2),
                (ConstSlot(5.0), SlotRef(4.0)),
            )
            Taped.store_tmp_value!(node, 1)
            @test node.tmp_slot[] == 5.0
            Taped.transfer_tmp_value!(node)
            @test node.ret_slot[] == 5.0
            Taped.store_tmp_value!(node, 2)
            @test node.tmp_slot[] == 4.0
            @test node.ret_slot[] == 5.0
            Taped.transfer_tmp_value!(node)
            @test node.ret_slot[] == 4.0
        end
        @testset "phi node with nothing in it" begin
            node = TypedPhiNode(SlotRef{Union{}}(), SlotRef{Union{}}(), (), ())
            Taped.store_tmp_value!(node, 1)
            Taped.transfer_tmp_value!(node)
        end
        @testset "phi node with undefined value" begin
            node = TypedPhiNode(
                SlotRef{Float64}(), SlotRef{Float64}(), (1, ), (SlotRef{Float64}(),)
            )
            Taped.store_tmp_value!(node, 1)
            Taped.transfer_tmp_value!(node)
        end
    end

    @testset "Unit tests for nodes and associated instructions" begin

        global __x_for_gref = 5.0
        global __y_for_gref::Float64 = 4.0

        @testset "ReturnNode" begin
            @testset "build_instruction: ReturnNode, $(_typeof(args))" for args in Any[
                (SlotRef(5.0), SlotRef{Float64}()),
                (SlotRef(4), SlotRef{Any}()),
                (ConstSlot(5), SlotRef{Int}()),
                (ConstSlot(5.0), SlotRef{Real}()),
                (ConstSlot(:hi), SlotRef{Symbol}()),
                (ConstSlot(:hi), SlotRef{Any}()),
                (TypedGlobalRef(GlobalRef(Main, :__x_for_gref)), SlotRef{Any}()),
                (ConstSlot(sin), SlotRef{typeof(sin)}()),
            ]
                val, ret_slot = args
                oc = build_inst(ReturnNode, ret_slot, val)
                @test oc isa Taped.Inst
                output = oc(0)
                @test output == -1
                @test ret_slot[] == val[]
            end
        end

        @testset "GotoNode $label" for label in Any[1, 2, 3, 4, 5]
            oc = build_inst(GotoNode, label)
            @test oc isa Taped.Inst
            @test oc(3) == label
        end

        global __global_bool = false
        @testset "GotoIfNot $cond" for cond in Any[
            SlotRef(true), SlotRef(false),
            ConstSlot(true), ConstSlot(false),
            SlotRef{Any}(true), SlotRef{Real}(false),
            ConstSlot{Any}(true), ConstSlot{Any}(false),
            TypedGlobalRef(GlobalRef(Main, :__global_bool)),
        ]
            oc = build_inst(GotoIfNot, cond, 1, 2)
            @test oc isa Taped.Inst
            @test oc(5) == (cond[] ? 1 : 2)
        end

        global __global_bool = true
        @testset "PiNode" for (input, out, prev_blk, next_blk) in Any[
            (SlotRef{Any}(5.0), SlotRef{Float64}(), 2, 3),
            (ConstSlot{Float64}(5.0), SlotRef{Float64}(), 2, 2),
            (TypedGlobalRef(GlobalRef(Main, :__global_bool)), ConstSlot(true), 2, 2)
        ]
            oc = build_inst(PiNode, input, out, next_blk)
            @test oc isa Taped.Inst
            @test oc(prev_blk) == next_blk
            @test out[] == input[]
        end

        global __x_for_gref = 5.0
        @testset "GlobalRef" for (out, x, next_blk) in Any[
            (SlotRef{Float64}(), TypedGlobalRef(Main, :__x_for_gref), 5),
            (SlotRef{typeof(sin)}(), ConstSlot(sin), 4),
        ]
            oc = build_inst(GlobalRef, x, out, next_blk)
            @test oc isa Taped.Inst
            @test oc(4) == next_blk
            @test out[] == x[]
        end

        @testset "QuoteNode and literals" for (x, out, next_blk) in Any[
            (ConstSlot(5), SlotRef{Int}(), 5),
        ]
            oc = build_inst(nothing, x, out, next_blk)
            @test oc isa Taped.Inst
            @test oc(1) == next_blk
            @test out[] == x[]
        end

        @testset "Val{:boundscheck}" begin
            val_ref = SlotRef{Bool}()
            oc = build_inst(Val(:boundscheck), val_ref, 3)
            @test oc isa Taped.Inst
            @test oc(5) == 3
            @test val_ref[] == true
        end

        global __int_output = 5
        @testset "Val{:call}" for (arg_slots, evaluator, val_slot, next_blk) in Any[
            ((ConstSlot(sin), SlotRef(5.0)), Taped._eval, SlotRef{Float64}(), 3),
            ((ConstSlot(*), SlotRef(4.0), ConstSlot(4.0)), Taped._eval, SlotRef{Any}(), 3),
            (
                (ConstSlot(+), ConstSlot(4), ConstSlot(5)),
                Taped._eval,
                TypedGlobalRef(Main, :__int_output),
                2,
            ),
            (
                (ConstSlot(getfield), SlotRef((5.0, 5)), ConstSlot(1)),
                Taped.get_evaluator(
                    Taped.MinimalCtx(),
                    Tuple{typeof(getfield), Tuple{Float64, Int}, Int},
                    nothing,
                    false,
                ),
                SlotRef{Float64}(),
                3,
            ),
        ]
            oc = build_inst(Val(:call), arg_slots, evaluator, val_slot, next_blk)
            @test oc isa Taped.Inst
            @test oc(0) == next_blk
            f, args... = map(getindex, arg_slots)
            @test val_slot[] == f(args...)
        end

        @testset "Val{:skipped_expression}" begin
            oc = build_inst(Val(:skipped_expression), 3)
            @test oc isa Taped.Inst
            @test oc(5) == 3
        end

        @testset "Val{:throw_undef_if_not}" begin
            @testset "defined" begin
                slot_to_check = SlotRef(5.0)
                oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
                @test oc isa Taped.Inst
                @test oc(0) == 2
            end
            @testset "undefined (non-isbits)" begin
                slot_to_check = SlotRef{Any}()
                oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
                @test oc isa Taped.Inst
                @test_throws ErrorException oc(3)
            end
            @testset "undefined (isbits)" begin
                slot_to_check = SlotRef{Float64}()
                oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
                @test oc isa Taped.Inst

                # a placeholder for failing to throw an ErrorException when evaluated
                @test_broken oc(5) == 1 
            end
        end
    end

    # Check that a suite of test cases run and give the correct answer.
    interp = Taped.TInterp()
    @testset "$(_typeof((f, x...)))" for (a, b, c, f, x...) in 
        TestResources.generate_test_functions()

        sig = _typeof((f, x...))
        @info "$sig"
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)

        # Verify correctness.
        @assert f(x...) == f(x...) # check that the primal runs
        x_cpy_1 = deepcopy(x)
        x_cpy_2 = deepcopy(x)
        @test has_equal_data(in_f(f, x_cpy_1...), f(x_cpy_2...))
        @test has_equal_data(x_cpy_1, x_cpy_2)
    end
end
