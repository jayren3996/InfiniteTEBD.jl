using Documenter
using InfiniteTEBD
using LinearAlgebra

DocMeta.setdocmeta!(InfiniteTEBD, :DocTestSetup, :(using InfiniteTEBD, LinearAlgebra); recursive=true)

const DOCS_SMOKE = get(ENV, "ITEBD_DOCS_SMOKE", "false") == "true"

makedocs(;
    sitename="InfiniteTEBD.jl",
    modules=[InfiniteTEBD],
    checkdocs=:exports,
    authors="jayren3996",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Overview" => "index.md",
        "Guide" => [
            "Getting Started" => "getting-started.md",
            "States and Canonical Form" => "imps.md",
            "Symmetric MPS" => "symmetries.md",
            "Time Evolution" => "time-evolution.md",
            "Observables" => "observables.md",
            "ScarFinder Workflow" => "scarfinder.md",
        ],
        "Reference" => [
            "API Reference" => "api.md",
        ],
    ],
    (DOCS_SMOKE ? (; remotes=nothing) : (;))...,
)

if !DOCS_SMOKE
    deploydocs(
        repo="github.com/jayren3996/InfiniteTEBD.jl.git",
        devbranch="master",
    )
end
