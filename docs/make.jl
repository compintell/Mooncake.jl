using Documenter, DocumenterCitations, Mooncake

DocMeta.setdocmeta!(
    Mooncake,
    :DocTestSetup,
    quote
        using Random, Mooncake
    end;
    recursive=true,
)

makedocs(
    sitename="Mooncake.jl",
    format=Documenter.HTML(;
        mathengine = Documenter.KaTeX(
            Dict(
                :macros => Dict(
                    "\\RR" => "\\mathbb{R}",
                ),
            )
        ),
    ),
    modules=[Mooncake],
    checkdocs=:none,
    plugins=[
        CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric),
    ],
    pages = [
        "Mooncake.jl" => "index.md",
        "Understanding Mooncake.jl" => [
            "Introduction" => "understanding_intro.md",
            "Algorithmic Differentiation" => "algorithmic_differentiation.md",
            "Mooncake.jl's Rule System" => "mathematical_interpretation.md",
        ],
        "Utilities" => [
            "Tools for Rules" => "tools_for_rules.md",
            "Debug Mode" => "debug_mode.md",
            "Debugging and MWEs" => "debugging_and_mwes.md",
        ],
        "Developer Documentation" => [
            "Running Tests Locally" => "running_tests_locally.md",
            "Developer Tools" => "developer_tools.md",
        ],
        "Known Limitations" => "known_limitations.md",
    ]
)

deploydocs(repo="github.com/compintell/Mooncake.jl.git", push_preview=true)
