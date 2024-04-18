include("front_matter.jl")

@testset "Tapir.jl" begin
    if test_group == "basic"
        include("utils.jl")
        include("tangents.jl")
        include("fwds_rvs_data.jl")
        include("codual.jl")
        include("safe_mode.jl")
        include("stack.jl")
        @testset "interpreter" begin
            include(joinpath("interpreter", "contexts.jl"))
            include(joinpath("interpreter", "abstract_interpretation.jl"))
            include(joinpath("interpreter", "bbcode.jl"))
            include(joinpath("interpreter", "ir_utils.jl"))
            include(joinpath("interpreter", "ir_normalisation.jl"))
            include(joinpath("interpreter", "s2s_reverse_mode_ad.jl"))
        end
    elseif test_group == "rrules"
        include("test_utils.jl")
        @testset "rrules" begin
            @info "avoiding_non_differentiable_code"
            include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
            @info "blas"
            include(joinpath("rrules", "blas.jl"))
            @info "builtins"
            include(joinpath("rrules", "builtins.jl"))
            @info "foreigncall"
            include(joinpath("rrules", "foreigncall.jl"))
            @info "lapack"
            include(joinpath("rrules", "lapack.jl"))
            @info "low_level_maths"
            include(joinpath("rrules", "low_level_maths.jl"))
            @info "misc"
            include(joinpath("rrules", "misc.jl"))
            @info "new"
            include(joinpath("rrules", "new.jl"))
        end
        include("chain_rules_macro.jl")
    elseif test_group == "integration_testing/misc"
        include(joinpath("integration_testing/", "misc.jl"))
        include(joinpath("integration_testing", "battery_tests.jl"))
    elseif test_group == "integration_testing/diff_tests"
        include(joinpath("integration_testing", "diff_tests.jl"))
    elseif test_group == "integration_testing/distributions"
        include(joinpath("integration_testing", "distributions.jl"))
    elseif test_group == "integration_testing/gp"
        include(joinpath("integration_testing", "gp.jl"))
    elseif test_group == "integration_testing/special_functions"
        include(joinpath("integration_testing", "special_functions.jl"))
    elseif test_group == "integration_testing/array"
        include(joinpath("integration_testing", "array.jl"))
    elseif test_group == "integration_testing/turing"
        include(joinpath("integration_testing", "turing.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
