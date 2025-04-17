# Reverse-Mode Design

## Compilation process

Last checked: 09/02/2025, Julia v1.10.8 / v1.11.3, Mooncake 0.4.83.

This brief informal note was largely written by Guillaume Dalle while learning how Mooncake's internals operate for reverse-mode, in order to be able to add forwards-mode AD.
It should help readers orient themsleves when first trying to understand Mooncake's internals.

Rule building is done statically, based on types. Some methods accept values, e.g.
```julia
build_rrule(args...; debug_mode=false)
```
but these simply extract the type of all arguments and call another method of `build_rrule`.

The action happens in [`s2s_reverse_mode_ad.jl`](https://github.com/compintell/Mooncake.jl/blob/main/src/interpreter/s2s_reverse_mode_ad.jl), in particular the following method:
```julia
build_rrule(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false)
```
`sig_or_mi` is either a signature, such as `Tuple{typeof(foo), Float64}`, or a `Core.MethodInstance`.
Signatures are extracted from `Core.MethodInstance`s as necessary.

If a signature has a custom rule (`Mooncake.is_primitive` returns `true`), we take it, otherwise we generate the IR and differentiate it.

The forward- and reverse-pass IRs are created by `generate_ir`.
The `OpaqueClosure` allows going back from the IR to a callable object. More precisely we use `MistyClosure` to store the associated IR.

The `Pullback` and `DerivedRule` structs are convenience wrappers for `MistyClosure`s with some bookkeeping.

Diving one level deeper, in the following method:

```julia
generate_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
```

The function `lookup_ir` calls `Core.Compiler.typeinf_ircode` on a method instance, which is a lower-level version of `Base.code_ircode`.

The IR considered is of type `IRCode`, which is different from the `CodeInfo` returned by `@code_typed`.
This format is obtained from `CodeInfo`, used to perform most optimizations in the Julia IR in the [evaluation pipeline](https://docs.julialang.org/en/v1/devdocs/eval/), then converted back to `CodeInfo`.

The function `normalise!` is a custom pass to modify `IRCode` and make some expressions nicer to work with.
The possible expressions one can encountered in lowered ASTs are documented [here](https://docs.julialang.org/en/v1/devdocs/ast/#Lowered-form).

Reverse-mode specific stuff: return type retrieval, `ADInfo`, `bbcode.jl`, `zero_like_rdata.jl`. The `BBCode` structure was a convenience for IR transformation.

Beyond the [`interpreter`](https://github.com/compintell/Mooncake.jl/blob/main/src/interpreter/) folder, check out [`tangents.jl`](https://github.com/compintell/Mooncake.jl/blob/main/src/tangents.jl) for forward mode.

`FData` and `RData` are not useful in forward mode, `Tangent` is the right representation.

For testing, `generate_test_functions` from [`test_resources.jl`](https://github.com/compintell/Mooncake.jl/blob/src/test_utils.jl) should all pass.
Recycle the functionality from reverse mode test utils.

To manipulate `IRCode`, check out the fields:

- `ir.argtypes` is the signature. Some are annotated with `Core.Const` to facilitate constant propagation for instance. Other annotations are `PartialStruct`, `Conditional`, `PartialTypeVar`. `Core.Compiler.widenconst` is used to extract types from these.
- `ir.stmts` is a `Core.Compiler.InstructionStream`. This represents a sequence of instructions via 5 vectors of the same length:
  - `stmts.stmt` is a vector of expressions (or other IR node types), see [AST docs](https://docs.julialang.org/en/v1/devdocs/ast/#Lowered-form)
  - `stmts.type` is a vector of types for the left-hand side of the assignment
  - three others
- `ir.cfg` is the Control Flow Graph of type `Core.Compiler.CFG`
- `ir.meta` is metadata, not important
- `ir.new_nodes` is an optimization buffer, not important
- `ir.sptypes` is for type parameters of the called function

We must maintain coherence between the various components of `IRCode` (especially `ir.stmts` and `ir.cfg`). That is the reason behind `BBCode`, to make coherence easier.
We can deduce the CFG from the statements but not the other way around: it's only composed of blocks of statement indices.
In forward mode we shouldn't have to modify anything but `ir.stmts`.
Do line by line transformation of the statements and then possibly refresh the CFG.

Example of line-by-line transformations are in `make_ad_stmts!`.
The `IRCode` nodes are not explicitly documented in <https://docs.julialang.org/en/v1/devdocs/ast/#Lowered-form> or <https://docs.julialang.org/en/v1/devdocs/ssair/#Main-SSA-data-structure>. Might need completion of official docs, but Mooncake docs in the meantime.

Inlining pass can prevent us from using high-level rules by inlining the function (e.g. unrolling a loop).
The contexts in [`interpreter/contexts.jl`](https://github.com/compintell/Mooncake.jl/blob/src/interpreter/contexts.jl) are `MinimalCtx` (necessary for AD to work) and `DefaultCtx` (ensure that we hit all of the rules).
Distinction between rules is not well maintained in Mooncake at the moment.
The function `is_primitive` defines whether we should recurse into the function during AD and break it into parts, or look for a rule.
If we define a rule we should set `is_primitive` to `true` for the corresponding function.

In [`interpreter/abstract_interpretation.jl`](https://github.com/compintell/Mooncake.jl/blob/src/interpreter/abstract_interpretation.jl) we interact with the Julia compiler.
The most important part is preventing the compiler from inlining.

The `MooncakeInterpreter` subtypes `Core.Compiler.AbstractInterpreter` to interpret Julia code.
There are also Cthulhu, Enzyme, JET interpreters.
Tells you how things get run.

For second order we will need to adapt IR lookup to misty closures.
