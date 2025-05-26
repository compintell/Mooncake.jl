mc_foo(x) = 2x

@testset "misty_closures" begin

    # Construct a sample MistyClosure.
    ir = Base.code_ircode_by_type(Tuple{typeof(mc_foo), Float64})[1][1]
    ir.argtypes[1] = Any
    mc = Mooncake.MistyClosure(ir)

    @testset "tangent interface etc" begin
        rng = StableRNG(123456)
        TestUtils.test_tangent_interface(rng, mc)
    end

    # fr = Mooncake.build_frule(Mooncake.get_interpreter(), mc)
    # display(fr.fwd_oc.ir[])
    # @show fr(zero_dual(mc_foo), zero_dual(5.0))
end
