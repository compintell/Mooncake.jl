using Documenter, DocumenterCitations, Tapir

DocMeta.setdocmeta!(
    Tapir,
    :DocTestSetup,
    quote
        using Tapir
    end,
)

makedocs(
    sitename="Tapir.jl",
    format=Documenter.HTML(;
        mathengine = Documenter.KaTeX(
            Dict(
                :macros => Dict(
                    "\\RR" => "\\mathbb{R}",
                ),
            )
        ),
    ),
    modules=[Tapir],
    checkdocs=:none,
    plugins=[
        CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric),
    ],
)

deploydocs(repo="github.com/withbayes/Tapir.jl.git", push_preview=true)
