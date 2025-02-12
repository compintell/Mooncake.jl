macro public(ex)
    @static if isdefined(Base, :ispublic)
        args = if ex isa Symbol
            (ex,)
        elseif Base.isexpr(ex, :tuple)
            ex.args
        else
            error("Misuse of `@public` macro on $ex")
        end
        esc(Expr(:public, args...))
    else
        nothing
    end
end
