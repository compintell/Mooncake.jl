include("front_matter.jl")

@testset "Mooncake.jl" begin
    if test_group == "aqua"
        Aqua.test_all(Mooncake)
    elseif test_group == "basic"
        include("utils.jl")
        include("tangents.jl")
        include("fwds_rvs_data.jl")
        include("codual.jl")
        include("debug_mode.jl")
        include("stack.jl")
        @testset "interpreter" begin
            include(joinpath("interpreter", "contexts.jl"))
            include(joinpath("interpreter", "abstract_interpretation.jl"))
            include(joinpath("interpreter", "bbcode.jl"))
            include(joinpath("interpreter", "ir_utils.jl"))
            include(joinpath("interpreter", "ir_normalisation.jl"))
            include(joinpath("interpreter", "zero_like_rdata.jl"))
            include(joinpath("interpreter", "s2s_reverse_mode_ad.jl"))
        end
        include("tools_for_rules.jl")
        include("interface.jl")
        include("config.jl")
        include("developer_tools.jl")
    elseif test_group == "rrules"
        include("test_utils.jl")
        @testset "rrules" begin
            @info "avoiding_non_differentiable_code"
            include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
            @info "blas"
            include(joinpath("rrules", "blas.jl"))
            @info "builtins"
            include(joinpath("rrules", "builtins.jl"))
            @info "fastmath"
            include(joinpath("rrules", "fastmath.jl"))
            @info "foreigncall"
            include(joinpath("rrules", "foreigncall.jl"))
            @info "iddict"
            include(joinpath("rrules", "iddict.jl"))
            @info "lapack"
            include(joinpath("rrules", "lapack.jl"))
            @info "linear_algebra"
            include(joinpath("rrules", "linear_algebra.jl"))
            @info "low_level_maths"
            include(joinpath("rrules", "low_level_maths.jl"))
            @info "misc"
            include(joinpath("rrules", "misc.jl"))
            @info "new"
            include(joinpath("rrules", "new.jl"))
            @info "tasks"
            include(joinpath("rrules", "tasks.jl"))
            @static if VERSION >= v"1.11.0-rc4"
                @info "memory"
                include(joinpath("rrules", "memory.jl"))
            end
        end
    elseif test_group == "gpu"
        include(joinpath("ext", "cuda", "cuda.jl"))
    elseif test_group == "ext/differentiation_interface"
        include(joinpath("ext", "differentiation_interface", "di.jl"))
    elseif test_group == "ext/dynamic_ppl"
        include(joinpath("ext", "dynamic_ppl", "dynamic_ppl.jl"))
    elseif test_group == "ext/logdensityproblemsad"
        include(joinpath("ext", "logdensityproblemsad", "logdensityproblemsad.jl"))
    elseif test_group == "ext/luxlib"
        include(joinpath("ext", "luxlib", "luxlib.jl"))
    elseif test_group == "ext/nnlib"
        include(joinpath("ext", "nnlib", "nnlib.jl"))
    elseif test_group == "ext/special_functions"
        include(joinpath("ext", "special_functions", "special_functions.jl"))
    elseif test_group == "integration_testing/array"
        include(joinpath("integration_testing", "array.jl"))
    elseif test_group == "integration_testing/bijectors"
        include(joinpath("integration_testing", "bijectors", "bijectors.jl"))
    elseif test_group == "integration_testing/diff_tests"
        include(joinpath("integration_testing", "diff_tests.jl"))
    elseif test_group == "integration_testing/distributions"
        include(joinpath("integration_testing", "distributions", "distributions.jl"))
    elseif test_group == "integration_testing/gp"
        include(joinpath("integration_testing", "gp", "gp.jl"))
    elseif test_group == "integration_testing/logexpfunctions"
        include(joinpath("integration_testing", "logexpfunctions", "logexpfunctions.jl"))
    elseif test_group == "integration_testing/lux"
        include(joinpath("integration_testing", "lux", "lux.jl"))
    elseif test_group == "integration_testing/misc"
        include(joinpath("integration_testing", "battery_tests.jl"))
    elseif test_group == "integration_testing/misc_abstract_array"
        include(joinpath("integration_testing", "misc_abstract_array.jl"))
    elseif test_group == "integration_testing/temporalgps"
        include(joinpath("integration_testing", "temporalgps", "temporalgps.jl"))
    elseif test_group == "integration_testing/turing"
        include(joinpath("integration_testing", "turing", "turing.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
