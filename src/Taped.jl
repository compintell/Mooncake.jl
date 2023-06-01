module Taped

using DiffRules, Distributions, FunctionWrappers, LinearAlgebra, Umlaut
import Umlaut: isprimitive, Frame, Tracer

using FunctionWrappers: FunctionWrapper

# Core functionality.
include("tracing.jl")

# Functions of tapes which don't output another tape.
include("is_pure.jl")

# Functions of tapes which output tapes.
include("vmap.jl")
include("forwards_mode_ad.jl")
include("reverse_mode_ad.jl")
include("logpdf.jl")
include("inplace.jl")
include("accelerate_tape.jl")

export primal, shadow

end
