is_pure(::F, x...) where {F} = false

# Built-ins.
for f in [
    # Built-ins.
    Core.apply_type, Core.arrayref, Core.arraysize, Core.compilerbarrier,
    Core.const_arrayref, Core.donotdelete, Core.finalizer, Core.get_binding_type,
    Core.ifelse, Core.sizeof, Core.svec, applicable, fieldtype, getfield, getglobal, isa,
    isdefined, nfields, tuple, typeassert, typeof,

    # Some basic maths for the sake of testing.
    sin, cos, tan,
]
    @eval is_pure(::typeof($f), x...) = true
end

for f in [sin, cos, tan]
    @eval is_pure(::typeof($f), ::Union{Float32, Float64}) = true
end

_val(op) = op
_val(op::Umlaut.AbstractOp) = op.val

is_pure(::Constant) = true
is_pure(::Input) = true
is_pure(c::Call) = is_pure(c.fn, map(_val, c.args)...)

is_pure(tape::Tape) = all(is_pure, tape.ops)
