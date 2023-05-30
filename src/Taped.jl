module Taped

using DiffRules, Distributions, Umlaut
import Umlaut: isprimitive, Frame, Tracer

# Core functionality.
include("tracing.jl")

# Functions of tapes which don't output another tape.
include("is_pure.jl")

# Functions of tapes which output tapes.
include("vmap.jl")
include("forwards_mode_ad.jl")
include("reverse_mode_ad.jl")
include("logpdf.jl")

end
