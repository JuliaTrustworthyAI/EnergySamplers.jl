using EnergySamplers
using Documenter

DocMeta.setdocmeta!(
  EnergySamplers,
  :DocTestSetup,
  :(using EnergySamplers);
  recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers

makedocs(;
  modules = [EnergySamplers],
  authors = "Patrick Altmeyer <p.altmeyer@tudelft.nl> and contributors",
  repo = "https://github.com/JuliaTrustworthyAI/EnergySamplers.jl/blob/{commit}{path}#{line}",
  sitename = "EnergySamplers.jl",
  format = Documenter.HTML(;
    canonical = "https://JuliaTrustworthyAI.github.io/EnergySamplers.jl",
  ),
  pages = [
    "index.md"
    [
      file for file in readdir(joinpath(@__DIR__, "src")) if
      file != "index.md" && splitext(file)[2] == ".md"
    ]
  ],
)

deploydocs(; repo = "github.com/JuliaTrustworthyAI/EnergySamplers.jl")
