using PlaceholderLikelihood
import Pkg; Pkg.add("Plots")
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
            "Plots" => "user_interface/plots.md",
            "Saving and Loading LikelihoodModels"=> "user_interface/saving_and_loading.md",
            "Simulated Coverage Checks" => 
                ["Parameter Confidence Intervals" => "user_interface/coverage/univariate_intervals.md"] ],
        "Internal Library" => 
            ["Common Functions" => "internal_library/common.md",
            "Initialisation" => "internal_library/initialisation.md",
            "Ellipse Functions" => "internal_library/ellipse_likelihood.md",
            "Univariate Functions" => "internal_library/univariate.md",
            "Bivariate Functions" => "internal_library/bivariate.md",
            "Dimensional Functions" => "internal_library/dimensional.md",
            "Prediction Functions" => "internal_library/predictions.md",
            "Plotting Functions" => "internal_library/plots.md"]
    ]
)

deploydocs(;
    repo="github.com/JoelTrent/PlaceholderLikelihood.jl",
    devbranch="main",
)
