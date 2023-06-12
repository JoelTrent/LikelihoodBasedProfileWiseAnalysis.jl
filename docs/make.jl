using PlaceholderLikelihood
using Plots
using Documenter

DocMeta.setdocmeta!(PlaceholderLikelihood, :DocTestSetup, :(using PlaceholderLikelihood); recursive=true)

makedocs(;
    modules=[PlaceholderLikelihood],
    authors="JoelTrent <79883375+JoelTrent@users.noreply.github.com> and contributors",
    repo="https://github.com/JoelTrent/PlaceholderLikelihood.jl/blob/{commit}{path}#{line}",
    sitename="PlaceholderLikelihood.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JoelTrent.github.io/PlaceholderLikelihood.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start" => "quick_start.md",
        "User Interface" => 
            ["Initialisation" => "user_interface/initialisation.md",
            "Parameter Profiles and Samples" => 
                ["Structs and Profile Types" => "user_interface/profiles_and_samples/profile_structs.md",
                "Univariate Profiles" => "user_interface/profiles_and_samples/univariate.md",
                "Bivariate Profiles" => "user_interface/profiles_and_samples/bivariate.md",
                "Dimensional Samples" => "user_interface/profiles_and_samples/dimensional.md"],
            "Predictions" => "user_interface/predictions.md",
            "Plots" => "user_interface/plots.md"],
        "Internal Library" => "internal_library.md"
    ]
)

deploydocs(;
    repo="github.com/JoelTrent/PlaceholderLikelihood.jl",
    devbranch="main",
)
