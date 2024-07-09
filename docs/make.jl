using LikelihoodBasedProfileWiseAnalysis
using Plots
using Documenter, DocumenterCitations

DocMeta.setdocmeta!(LikelihoodBasedProfileWiseAnalysis, :DocTestSetup, :(using LikelihoodBasedProfileWiseAnalysis); recursive=true)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

makedocs(;
    modules=[LikelihoodBasedProfileWiseAnalysis],
    authors="JoelTrent <79883375+JoelTrent@users.noreply.github.com> and contributors",
    sitename="LikelihoodBasedProfileWiseAnalysis.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JoelTrent.github.io/LikelihoodBasedProfileWiseAnalysis.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Background" => 
            ["Motivation" => "workflow_motivation.md", 
            "Formulation" => "workflow_formulation.md"],
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
            "Timing and Profiling" => "user_interface/timing_and_profiling.md",
            "Simulated Coverage Checks" => 
                ["Parameter Confidence Intervals" => "user_interface/coverage/univariate_intervals.md",
                "Bivariate Parameter Confidence Boundaries" => "user_interface/coverage/bivariate_boundaries.md",
                "Predictions and Realisations" => "user_interface/coverage/predictions_and_realisations.md"]],
        "Examples" => ["Initial Setup" => "examples/index.md",
                        "Logistic Model" => "examples/logistic.md",
                        "Lotka-Volterra Model" => "examples/lotka-volterra.md",
                        "Two-Species Logistic Model"=> "examples/two-species_logistic.md",
                        "Gaussian Approximation of a Binomial Distribution"=> "examples/binomial_normal_approximation.md",
                        "Function Evaluation Timing - Logistic Model" => "examples/logistic_timing_estimates.md"],
        "Internal Library" => 
            ["Common Functions" => "internal_library/common.md",
            "Initialisation Internal" => "internal_library/initialisation.md",
            "Ellipse Functions" => "internal_library/ellipse_likelihood.md",
            "Univariate Functions" => "internal_library/univariate.md",
            "Bivariate Functions" => "internal_library/bivariate.md",
            "Dimensional Functions" => "internal_library/dimensional.md",
            "Prediction Functions" => "internal_library/predictions.md",
            "Plotting Functions" => "internal_library/plots.md",
            "Coverage Functions" => "internal_library/coverage.md"],
        "References" => "references.md"
    ],
    plugins=[bib],
    warnonly=true
)

deploydocs(;
    repo="github.com/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl",
    devbranch="main",
)
