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
            joinpath("understanding_mooncake", "introduction.md"),
            joinpath("understanding_mooncake", "algorithmic_differentiation.md"),
            joinpath("understanding_mooncake", "rule_system.md"),
        ],
        "Utilities" => [
            joinpath("utilities", "tools_for_rules.md"),
            joinpath("utilities", "debug_mode.md"),
            joinpath("utilities", "debugging_and_mwes.md"),
        ],
        "Developer Documentation" => [
            joinpath("developer_documentation", "running_tests_locally.md"),
            joinpath("developer_documentation", "developer_tools.md"),
            joinpath("developer_documentation", "internal_docstrings.md"),
        ],
        "known_limitations.md",
    ]
)

deploydocs(repo="github.com/compintell/Mooncake.jl.git", push_preview=true)
