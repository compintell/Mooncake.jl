@testset "DynamicPPLTapirExt" begin
    Tapir.TestUtils.test_rule(
        Xoshiro(123456), DynamicPPL.istrans, DynamicPPL.VarInfo();
        perf_flag=:none,
        interface_only=true,
        is_primitive=true,
        interp=Tapir.TapirInterpreter(),
    )
end
