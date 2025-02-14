# The purpose of the code in this file is to provide a work-around for the Julia compiler
# bug discussed in https://github.com/compintell/Mooncake.jl/issues/319 . You do not need to
# understand it in order to understand Mooncake. I (Will) would recommend against spending
# any time at all reading / understanding this file unless you are actively working on this
# issue, and find it helpful.
#
# As soon as patch versions of Julia (both LTS and Release) are made which contain a fix for
# the bug underlying the above issue, this code can and should be removed, and the minimum
# versions of Julia which Mooncake supports bumped.
#
# The only place in which this code seeps into Mooncake.jl code is in Mooncake.optimise_ir!,
# located in src/interpreter/ir_utils.jl . In particular, we replace the `local_interp`
# variable with the `AbstractInterpreter`.
#
# The work around:
# We define a new `AbstractInterpreter` which wraps around the `Compiler.NativeInterpreter`.
# This makes it possible to add methods to various functions in `Compiler`, thereby enabling
# us to insert the bug fixes.

