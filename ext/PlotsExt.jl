module PlotsExt

    using LikelihoodBasedProfileWiseAnalysis
    using Plots, LaTeXStrings
    using DataFrames, StatsBase

    """
        profilecolor(profile_type::Union{AbstractProfileType, AbstractSampleType})

    Colors of profile types and sample types as integers between 1 and 6.
    """
    function profilecolor(profile_type::Union{AbstractProfileType, AbstractSampleType})
        if profile_type isa EllipseApproxAnalytical
            return 1
        elseif profile_type isa EllipseApprox
            return 2
        elseif profile_type isa UniformGridSamples
            return 4
        elseif profile_type isa UniformRandomSamples
            return 5
        elseif profile_type isa LatinHypercubeSamples
            return 6
        end
        return 3
    end

    """
        profile1Dlinestyle(profile_type::AbstractProfileType)

    Linestyle of each `profile_type`. [`EllipseApproxAnalytical`](@ref) - :dash, [`EllipseApprox`](@ref) - :dashdot, [`LogLikelihood`](@ref) - :solid, 
    """
    function profile1Dlinestyle(profile_type::AbstractProfileType)
        if profile_type isa EllipseApproxAnalytical
            return :dash
        elseif profile_type isa EllipseApprox
            return :dashdot
        end
        return :solid
    end

    """
        profile2Dmarkershape(profile_type::Union{AbstractProfileType, AbstractSampleType}, on_boundary::Bool)

    Marker shapes of bivariate profiles depending on profile type or sample type and whether or not the point is on the profile's boundary.
    """
    function profile2Dmarkershape(profile_type::Union{AbstractProfileType, AbstractSampleType}, on_boundary::Bool)
        if profile_type isa EllipseApproxAnalytical
            return on_boundary ? :diamond : :hexagon
        elseif profile_type isa EllipseApprox
            return on_boundary ? :utriangle : :dtriangle
        elseif profile_type isa AbstractSampleType
            return on_boundary ? :ltriangle : :rtriangle
        end
        return on_boundary ? :circle : :rect
    end

    """
        θs_to_plot_typeconversion(model::LikelihoodModel, θs_to_plot::Union{Vector{<:Symbol}, Vector{<:Int64}})

    Converts `θs_to_plot` to integer index values with [`LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices`](@ref) if they are supplied as a vector of symbols.
    """
    function θs_to_plot_typeconversion(model::LikelihoodModel,
                                        θs_to_plot::Union{Vector{<:Symbol}, Vector{<:Int64}})
        if θs_to_plot isa Vector{<:Symbol}
            return LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices(model, θs_to_plot)
        end
        return θs_to_plot
    end

    """
        θcombinations_to_plot_typeconversion(model::LikelihoodModel,
            θcombinations_to_plot::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}, Vector{Vector{Int}}, Vector{Tuple{Int,Int}}})

    Converts `θcombinations_to_plot` to a vector of tuples of integer index values with [`LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices`](@ref) if they are supplied as a vector of vectors/tuples containing symbols. The index elements in each vector/tuple will be sorted in ascending order. 
    """
    function θcombinations_to_plot_typeconversion(model::LikelihoodModel,
                                                    θcombinations_to_plot::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}, Vector{Vector{Int}}, Vector{Tuple{Int,Int}}})
        if θcombinations_to_plot isa Vector{Vector{Symbol}} || θcombinations_to_plot isa Vector{Tuple{Symbol, Symbol}}
            return [tuple(sort(arr)...) for arr in LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices(model, θcombinations_to_plot)]
        end

        if θcombinations_to_plot isa Vector{Vector{Int}}
            return [tuple(sort(arr)...) for arr in θcombinations_to_plot] 
        end
        return θcombinations_to_plot
    end

    # plotting functions #################

    """
        plot1Dprofile!(plt, parRange, parProfile, label="profile"; kwargs...)

    Plots a univariate profile at ψ locations `parRange`, with normalised profile log-likelihood function values `parProfile`.
    """
    function plot1Dprofile!(plt, parRange, parProfile, label="profile"; kwargs...)
        plot!(plt, parRange, parProfile, lw=3, label=label; kwargs...)
        return plt
    end

    """
        addMLEandLLstar!(plt, llstar, parMLE, MLE_color, llstar_color; kwargs...)

    On a univariate profile, adds the MLE location as a vertical line and the asymptotic confidence threshold as a horizontal line. The intersection between the horizontal asymptotic confidence threshold and the univariate profile gives the location of the corresponding confidence interval for that parameter. 
    """
    function addMLEandLLstar!(plt, llstar, parMLE, MLE_color, llstar_color; kwargs...)
        vline!(plt, [parMLE], lw=3, color=MLE_color, label="MLE point", linestyle=:dot)
        hline!(plt, [llstar], lw=3, color=llstar_color, label=L"\ell_{c}", linestyle=:dot; kwargs...)
        return plt
    end

    """
        plot2Dboundary!(plt, parBoundarySamples, label="boundary"; use_lines=false, kwargs...)

    Plots the boundary of a bivariate profile at 2D locations `parBoundarySamples`. If `use_lines=false` then it will use a scatter plot, otherwise it will connect the boundary in the order of the `parBoundarySamples`.
    """
    function plot2Dboundary!(plt, parBoundarySamples, label="boundary"; use_lines=false, kwargs...)
        
        plot_func! = use_lines ? plot! : scatter!
        plot_func!(plt, parBoundarySamples[1,:], parBoundarySamples[2,:], 
                                msw=0, ms=5,
                                label=label;
                                kwargs...)
        return plt
    end

    """
        addMLE!(plt, parMLEs; kwargs...)

    On a bivariate profile, adds the MLE location as a scatter point.
    """
    function addMLE!(plt, parMLEs; kwargs...)
        scatter!(plt, [parMLEs[1]], [parMLEs[2]],
                markershape=:circle,
                msw=0, ms=5,
                label="MLE point"; kwargs...)
        return plt
    end

    # Predictions

    """
        plotprediction!(plt, t, predictions, extrema, linealpha, layout; extremacolor=:red, kwargs...)

    Plots the profile-wise confidence set for the model trajectory using the same format as in the profile-wise analysis workflow paper [simpsonprofilewise2023](@cite). The extrema of the profile-wise trajectory confidence set is labelled as approximate profile-wise simultaneous confidence bands (SCBs).
    """
    function plotprediction!(plt, t, predictions, extrema, linealpha, layout; extremacolor=:red, kwargs...)
        if layout > 1
            for i in 1:(layout-1)
                plot!(plt[i], t, predictions[:,:,i], color=:grey, linealpha=linealpha; label=hcat("Profile-wise predictions", fill("", 1, size(predictions, 2)-1)), kwargs...)
                plot!(plt[i], t, extrema[:,:,i], lw=3, color=extremacolor, label=["Profile-wise SCBs (≈)" ""], legend=false)
            end
            plot!(plt[layout], t, predictions[:,:,layout], color=:grey, linealpha=linealpha; label=hcat("Profile-wise predictions", fill("", 1, size(predictions, 2)-1)), kwargs...)
            plot!(plt[layout], t, extrema[:,:,layout], lw=3, color=extremacolor, label=["Profile-wise SCBs (≈)" ""])
            return plt
        end    
        plot!(plt, t, predictions, color=:grey, linealpha=linealpha; label=hcat("Profile-wise predictions", fill("", 1, size(predictions, 2)-1)), kwargs...)
        plot!(plt, t, extrema, lw=3, color=extremacolor, label=["Profile-wise SCBs (≈)" ""])
        return plt
    end

    """
        plotrealisation!(plt, t, extrema, linealpha, layout; extremacolor=:red, kwargs...)

    Plots the extrema of the profile-wise reference tolerance set for the (1-δ) population reference set. The (1-δ) population reference set refers to a set containing the (1-δ) reference region of the population at each time point; i.e. the smallest region at each time point which contains (1-δ) of possible realisations. The extrema of the profile-wise reference tolerance set is labelled as approximate profile-wise simultaneous reference tolerance bands (SRTBs).
    """
    function plotrealisation!(plt, t, extrema, linealpha, layout; extremacolor=:red, kwargs...)
        if layout > 1
            for i in 1:(layout-1)
                plot!(plt[i], t, extrema[:,:,i]; lw=3, color=extremacolor, label=["Profile-wise SRTBs (≈)" ""], legend=false, kwargs...)
            end
            plot!(plt[layout], t, extrema[:,:,layout]; lw=3, color=extremacolor, label=["Profile-wise SRTBs (≈)" ""], kwargs...)
            return plt
        end    
        plot!(plt, t, extrema; lw=3, color=extremacolor, label=["Profile-wise SRTBs (≈)" ""], kwargs...)
        return plt
    end

    """
        add_yMLE!(plt, t, yMLE, layout; kwargs...)

    Adds the model trajectory obtained from simulating the model at times `t` using the maximum likelihood estimate for parameters. 
    """
    function add_yMLE!(plt, t, yMLE, layout; kwargs...)
        if layout > 1
            for i in 1:layout
                plot!(plt[i], t, yMLE[:,i], lw=3, color=:turquoise1, label="MLE"; kwargs...)
            end
            return plt
        end
        plot!(plt, t, yMLE, lw=3, color=:turquoise1, label="MLE"; kwargs...)
        return plt
    end


    """
        add_extrema!(plt, t, extrema, layout; extremacolor=:gold, label=["Sampled SCBs (≈)" ""])

    Adds additional extrema to plots for the model trajectory and population reference sets. Typically for comparing the extrema of profile-wise predictions sets from profiles with sets from dimensional samples (often the full parameter confidence set).
    """
    function add_extrema!(plt, t, extrema, layout; extremacolor=:gold, label=["Sampled SCBs (≈)" ""])
        if layout > 1
            for i in 1:layout
                plot!(plt[i], t, extrema[:,:,i], lw=3, color=extremacolor, label=label)
            end
            return plt
        end
        plot!(plt, t, extrema, lw=3, color=extremacolor, label=label)
        return plt
    end

    """
        plot_univariate_profiles(model::LikelihoodModel,
            xlim_scaler::Real=0.2,
            ylim_scaler::Real=0.2;
            θs_to_plot::Vector=Int[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[], 
            num_points_in_interval::Int=0,
            palette_to_use::Symbol=:Paired_6, 
            kwargs...)

    Returns a vector of plots of univariate profiles contained with the `model` struct that meet the requirements of the univariate method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). 

    The profiles plotted are based on the specified `θs_to_plot`, `confidence_levels`, `dofs` and `profile_types`. By default, will plot all univariate profiles generated.

    If `num_points_in_interval` is greater than 0 then [`get_points_in_intervals!`](@ref) will be called - use to obtain smoother profile plots.

    `xlim_scaler` and `ylim_scaler` are used to uniformly push the `xlimits` and `ylimits` away from the location of the confidence interval - if they are zero, then the confidence interval gives the location of the `xlimits` and the lower of the `ylimits`. If they are `1` then the corresponding limits have a range 100% wider than the confidence interval.
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_univariate_profiles(model::LikelihoodModel,
                                        xlim_scaler::Real=0.2,
                                        ylim_scaler::Real=0.2;
                                        θs_to_plot::Vector=Int[],
                                        confidence_levels::Vector{<:Float64}=Float64[],
                                        dofs::Vector{<:Int}=Int[],
                                        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[], 
                                        num_points_in_interval::Int=0,
                                        palette_to_use::Symbol=:Paired_6, 
                                        kwargs...)

        if num_points_in_interval > 0
            get_points_in_intervals!(model, num_points_in_interval, 
                                    confidence_levels=confidence_levels, 
                                    dofs=dofs,
                                    profile_types=profile_types)
        end
        
        θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)

        sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, θs_to_plot, confidence_levels, dofs, profile_types)

        if nrow(sub_df) < 1
            return nothing
        end
        
        color_palette = palette(palette_to_use)
        profile_plots = [plot() for _ in 1:nrow(sub_df)]

        for i in 1:nrow(sub_df)

            row = @view(sub_df[i,:])
            interval = LikelihoodBasedProfileWiseAnalysis.get_uni_confidence_interval_points(model, row.row_ind)
            boundary_col_indices = interval.boundary_col_indices

            llstar = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(model, row.conf_level, EllipseApprox(), row.dof)
            parMLE = model.core.θmle[row.θindex]
            θname = model.core.θnames[row.θindex]
            
            x_range = interval.points[row.θindex, boundary_col_indices[2]] - interval.points[row.θindex, boundary_col_indices[1]]
            
            plot1Dprofile!(profile_plots[i], interval.points[row.θindex, :], interval.ll; 
                xlims=[interval.points[row.θindex,  boundary_col_indices[1]]-x_range*xlim_scaler, 
                interval.points[row.θindex,  boundary_col_indices[2]]+x_range*xlim_scaler],
                color=color_palette[profilecolor(row.profile_type)])
            
            addMLEandLLstar!(profile_plots[i], llstar, parMLE, color_palette[end-1], color_palette[end]; 
                            xlabel=string(θname), ylabel=L"\hat{\ell}_{p}", 
                            ylims=[llstar + llstar*ylim_scaler, 0.1],
                            title=string("Profile type: ", row.profile_type, 
                                            "\nConfidence level: ", row.conf_level, ", dof: ", row.dof),
                            titlefontsize=10, kwargs...)
            
        end

        return profile_plots
    end

    """
        plot_univariate_profiles_comparison(model::LikelihoodModel, 
            xlim_scaler::Real=0.2,
            ylim_scaler::Real=0.2;
            θs_to_plot::Vector=Int[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[], 
            num_points_in_interval::Int=0,
            palette_to_use::Symbol=:Paired_6,
            label_only_lines::Bool=false,
            kwargs...)

    Returns a vector of comparison plots of univariate profiles contained with the `model` struct that meet the requirements of the univariate method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). Comparisons are between `profile_types` at the same `confidence_level` and `dof` for a given parameter.

    The profiles plotted are based on the specified `θs_to_plot`, `confidence_levels`, `dofs` and `profile_types`. By default, will plot all univariate profiles generated.

    If `num_points_in_interval` is greater than 0 then [`get_points_in_intervals!`](@ref) will be called - use to obtain smoother profile plots.

    If `label_only_lines=true` then only the vertical and horizontal MLE point and confidence threshold lines will be labelled in the legend. Otherwise, profiles will be labelled by their `profile_type`.

    `xlim_scaler` and `ylim_scaler` are used to uniformly push the `xlimits` and `ylimits` away from the location of the confidence interval - if they are zero, then the confidence interval gives the location of the `xlimits` and the lower of the `ylimits`. If they are `1` then the corresponding limits have a range 100% wider than the confidence interval.
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_univariate_profiles_comparison(model::LikelihoodModel, 
                                        xlim_scaler::Real=0.2,
                                        ylim_scaler::Real=0.2;
                                        θs_to_plot::Vector=Int[],
                                        confidence_levels::Vector{<:Float64}=Float64[],
                                        dofs::Vector{<:Int}=Int[],
                                        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[], 
                                        num_points_in_interval::Int=0,
                                        palette_to_use::Symbol=:Paired_6,
                                        label_only_lines::Bool=false,
                                        kwargs...)

        if num_points_in_interval > 0
            get_points_in_intervals!(model, num_points_in_interval, 
                                    confidence_levels=confidence_levels, 
                                    profile_types=profile_types)
        end

        θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)

        sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, θs_to_plot, confidence_levels, dofs, profile_types)

        if nrow(sub_df) < 1
            return nothing
        end

        if isempty(confidence_levels)
            confidence_levels = unique(sub_df.conf_level)
        end
        if isempty(dofs)
            dofs = unique(sub_df.dof)
        end

        color_palette = palette(palette_to_use)
        profile_plots = [plot()]
        plot_i=1

        row_subset = trues(nrow(sub_df))

        for θi in 1:model.core.num_pars
            for dof in dofs
                for confidence_level in confidence_levels

                    row_subset .= (sub_df.θindex .== θi) .& (sub_df.dof .== dof) .& (sub_df.conf_level .== confidence_level)
                    conf_df = @view(sub_df[row_subset, :])

                    if nrow(conf_df) > 1
                        if plot_i > 1
                            append!(profile_plots, [plot()])
                        end
                        llstar = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(model, confidence_level, EllipseApprox(), dof)
                        parMLE = model.core.θmle[θi]
                        θname = model.core.θnames[θi]

                        xlims = zeros(2)

                        for i in 1:nrow(conf_df)

                            row = @view(conf_df[i,:])
                            interval = LikelihoodBasedProfileWiseAnalysis.get_uni_confidence_interval_points(model, row.row_ind)
                            boundary_col_indices = interval.boundary_col_indices
                            
                            x_range = interval.points[row.θindex, boundary_col_indices[2]] - interval.points[row.θindex, boundary_col_indices[1]]

                            if i == 1
                                xlims .= [interval.points[row.θindex, boundary_col_indices[1]] - x_range*xlim_scaler, 
                                    interval.points[row.θindex, boundary_col_indices[2]] + x_range*xlim_scaler]
                            else
                                xlims[1] = min(xlims[1], interval.points[row.θindex, boundary_col_indices[1]] - x_range*xlim_scaler) 
                                xlims[2] = max(xlims[2], interval.points[row.θindex, boundary_col_indices[2]] + x_range*xlim_scaler)
                            end
                            
                            plot1Dprofile!(profile_plots[plot_i], interval.points[row.θindex, :], interval.ll; 
                                            label = label_only_lines ? "" : string(row.profile_type),
                                            linestyle=profile1Dlinestyle(row.profile_type),
                                            color=color_palette[profilecolor(row.profile_type)])
                        end

                        addMLEandLLstar!(profile_plots[plot_i], llstar, parMLE, color_palette[end-1], color_palette[end]; 
                                        xlabel=string(θname), ylabel=L"\hat{\ell}_{p}", 
                                        xlims=xlims,
                                        ylims=[llstar + llstar*ylim_scaler, 0.1],
                                        title=string("Confidence level: ", confidence_level, ", dof: ", dof),
                                        titlefontsize=10, kwargs...)

                        plot_i+=1
                    end
                end
            end
        end

        return profile_plots
    end

    """
        plot_bivariate_profiles(model::LikelihoodModel,
            xlim_scaler::Real=0.2,
            ylim_scaler::Real=0.2;
            for_dim_samples::Bool=false,
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
            palette_to_use::Symbol=:Paired_6,
            include_internal_points::Bool=true,
            max_internal_points::Int=1000,
            markeralpha=1.0,
            kwargs...)

    Returns a vector of plots of bivariate profiles contained with the `model` struct that meet the requirements of the bivariate method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments).

    The profiles plotted are based on the specified `θcombinations_to_plot`, `confidence_levels`, `dofs`, `profile_types`, `methods` and `sample_types`. By default, will plot all bivariate profiles generated. If `for_dim_samples=false` it will plot bivariate profiles generated by an [`AbstractBivariateMethod`](@ref). Otherwise, it will plot bivariate profiles generated by an [`AbstractSampleType`](@ref).

    If `include_internal_points=true` then points inside the boundary up to `max_internal_points` will be plotted (these are chosen randomly). Otherwise, only the boundary of the profile will be plotted. If plotting bivariate profiles from an [`AbstractSampleType`](@ref) this boundary will be estimated using [`LikelihoodBasedProfileWiseAnalysis.bivariate_concave_hull`](@ref).

    `xlim_scaler` and `ylim_scaler` are used to uniformly push the `xlimits` and `ylimits` away from the location of the confidence boundary - if they are zero, then the extrema of the confidence boundary gives the location of the `xlimits` and the `ylimits`. If they are `1` then the corresponding limits have a range 100% wider than the extrema of the confidence boundary.
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_bivariate_profiles(model::LikelihoodModel,
                                        xlim_scaler::Real=0.2,
                                        ylim_scaler::Real=0.2;
                                        for_dim_samples::Bool=false,
                                        θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                        confidence_levels::Vector{<:Float64}=Float64[],
                                        dofs::Vector{<:Int}=Int[],
                                        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                        sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                        palette_to_use::Symbol=:Paired_6,
                                        include_internal_points::Bool=true,
                                        max_internal_points::Int=1000,
                                        markeralpha=1.0,
                                        kwargs...)

        max_internal_points = max(1, max_internal_points)
        if for_dim_samples
            sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_levels, dofs, sample_types, 
                                        sample_dimension=2)
        else
            θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
                                            
            sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, θcombinations_to_plot, confidence_levels,
                                        dofs, profile_types, methods)
        end

        if nrow(sub_df) < 1
            return nothing
        end
        
        color_palette = palette(palette_to_use)
        profile_plots = [plot() for _ in 1:nrow(sub_df)]
        
        for i in 1:nrow(sub_df)

            row = @view(sub_df[i,:])
            θindices = zeros(Int,2); 
            for j in 1:2; θindices[j] = row.θindices[j] end
            
            if for_dim_samples
                boundary = LikelihoodBasedProfileWiseAnalysis.bivariate_concave_hull(model.dim_samples_dict[row.row_ind], θindices, 1.0, 1.0, 
                    LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(model, row.conf_level, EllipseApproxAnalytical(), row.dof), row.sample_type)
                profile_type=row.sample_type

                title=string("Sample type: ", profile_type, 
                            "\nConfidence level: ", row.conf_level, ", dof: ", row.dof)
            else
                full_boundary = model.biv_profiles_dict[row.row_ind].confidence_boundary
                boundary = @view(full_boundary[θindices, :])
                profile_type = row.profile_type

                title=string("Profile type: ", profile_type, 
                            "\nMethod: ", row.method,
                            "\nConfidence level: ", row.conf_level, ", dof: ", row.dof)
            end
            
            parMLEs = model.core.θmle[θindices]
            θnames = model.core.θnames[θindices]
            
            min_vals = minimum(boundary, dims=2)
            max_vals = maximum(boundary, dims=2)
            ranges = max_vals .- min_vals
            
            plot2Dboundary!(profile_plots[i], boundary, 
                                use_lines=for_dim_samples,
                                markershape=profile2Dmarkershape(profile_type, true),
                                markercolor=color_palette[profilecolor(profile_type)],
                                linecolor=color_palette[profilecolor(profile_type)],
                                markeralpha=markeralpha,
                                linealpha=markeralpha)

            if include_internal_points && (for_dim_samples || !row.not_evaluated_internal_points)
                internal_points = for_dim_samples ? @view(model.dim_samples_dict[row.row_ind].points[θindices, :]) : @view(model.biv_profiles_dict[row.row_ind].internal_points.points[θindices, :])
                
                num_internal_points = size(internal_points, 2)

                internal_points = @view(internal_points[:, sample(1:num_internal_points, min(max_internal_points, num_internal_points), replace=false, ordered=true)])
                plot2Dboundary!(profile_plots[i], 
                                internal_points,
                                "internal points", 
                                use_lines=false,
                                markershape=profile2Dmarkershape(profile_type, false), 
                                markercolor=color_palette[profilecolor(profile_type)], 
                                linecolor=color_palette[profilecolor(profile_type)],
                                markeralpha=markeralpha*0.4,
                                linealpha=markeralpha*0.5)
            end

            addMLE!(profile_plots[i], parMLEs; 
                markercolor=color_palette[end],
                xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                        max_vals[1]+ranges[1]*xlim_scaler],
                ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                max_vals[2]+ranges[2]*ylim_scaler],
                title=title,
                titlefontsize=10, kwargs...)
        end

        return profile_plots
    end

    """
        plot_bivariate_profiles_iterativeboundary_gif(model::LikelihoodModel,
            xlim_scaler::Real=0.2,
            ylim_scaler::Real=0.2;
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
            palette_to_use::Symbol=:Paired_6,
            save_as_separate_plots::Bool=false,
            markeralpha=1.0,
            save_folder=nothing,
            kwargs...)

    Saves a gif of the boundary of bivariate profiles generated using [`IterativeBoundaryMethod`](@ref) in `save_folder` that also meet the requirements of the bivariate method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments).

    The profiles plotted are based on the specified `θcombinations_to_plot`, `confidence_levels`, `dofs` and `profile_types`.

    `xlim_scaler` and `ylim_scaler` are used to uniformly push the `xlimits` and `ylimits` away from the location of the final confidence boundary - if they are zero, then the extrema of the confidence boundary gives the location of the `xlimits` and the `ylimits`. If they are `1` then the corresponding limits have a range 100% wider than the extrema of the confidence boundary. 

    If `save_as_separate_plots=true` then alongside the saved gif, each frame of the gif will also be saved as a `.png`.
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_bivariate_profiles_iterativeboundary_gif(model::LikelihoodModel,
                                        xlim_scaler::Real=0.2,
                                        ylim_scaler::Real=0.2;
                                        θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                        confidence_levels::Vector{<:Float64}=Float64[],
                                        dofs::Vector{<:Int}=Int[],
                                        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                        palette_to_use::Symbol=:Paired_6,
                                        save_as_separate_plots::Bool=false,
                                        # include_internal_points::Bool=true,
                                        markeralpha=1.0,
                                        save_folder=nothing,
                                        kwargs...)

        methods = [IterativeBoundaryMethod(1,1,1)]

        θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
                                        
        sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, θcombinations_to_plot, confidence_levels,
                                    dofs, profile_types, methods)

        if nrow(sub_df) < 1
            return nothing
        end
        
        color_palette = palette(palette_to_use)
        
        for i in 1:nrow(sub_df)

            row = @view(sub_df[i,:])
            θindices = zeros(Int,2); 
            for j in 1:2; θindices[j] = row.θindices[j] end
            
            full_boundary = model.biv_profiles_dict[row.row_ind].confidence_boundary
            boundary = @view(full_boundary[θindices, :])
            profile_type = row.profile_type

            title=string("Profile type: ", profile_type, 
                        "\nMethod: ", row.method,
                        "\nConfidence level: ", row.conf_level, ", dof: ", row.dof)
            
            parMLEs = model.core.θmle[θindices]
            θnames = model.core.θnames[θindices]
            
            min_vals = minimum(boundary, dims=2)
            max_vals = maximum(boundary, dims=2)
            ranges = max_vals .- min_vals

            plt = plot(dpi=200)

            addMLE!(plt, parMLEs; 
                markercolor=color_palette[end],
                xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                        max_vals[1]+ranges[1]*xlim_scaler],
                ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                max_vals[2]+ranges[2]*ylim_scaler],
                title=title,
                titlefontsize=10, kwargs...)

            plot2Dboundary!(plt, @view(boundary[:,1:row.method.initial_num_points-1]), 
                            use_lines=false,
                            markershape=profile2Dmarkershape(profile_type, true),
                            markercolor=color_palette[profilecolor(profile_type)],
                            # linecolor=color_palette[profilecolor(profile_type)],
                            markeralpha=markeralpha
                            # linealpha=markeralpha
                            )
            
            anim = @animate for k in (row.method.initial_num_points):row.num_points
                plot2Dboundary!(plt, @view(boundary[:,k]), 
                                use_lines=false,
                                label=nothing,
                                markershape=profile2Dmarkershape(profile_type, true),
                                markercolor=color_palette[profilecolor(profile_type)],
                                # linecolor=color_palette[profilecolor(profile_type)],
                                markeralpha=markeralpha
                                # linealpha=markeralpha
                                )
                if save_as_separate_plots
                    name=string(k)*".png"
                    savefig(plt, isnothing(save_folder) ? name : joinpath(save_folder, name))
                end
            end

            filename = string("iterative_boundary_", θnames[1],"_",θnames[2], ".gif")
            save_location = isnothing(save_folder) ? filename : joinpath(save_folder, filename) 

            gif(anim, save_location, fps=15)

            # if include_internal_points && !row.not_evaluated_internal_point
            #     internal_points = for_dim_samples ? @view(model.dim_samples_dict[row.row_ind].points[θindices, :]) : @view(model.biv_profiles_dict[row.row_ind].internal_points.points[θindices, :])
                
            #     num_internal_points = size(internal_points, 2)

            #     internal_points = @view(internal_points[:, sample(1:num_internal_points, min(1000, num_internal_points), replace=false, ordered=true)])
            #     plot2Dboundary!(profile_plots[i], 
            #                     internal_points,
            #                     "internal points", 
            #                     use_lines=false,
            #                     markershape=profile2Dmarkershape(profile_type, false), 
            #                     markercolor=color_palette[profilecolor(profile_type)], 
            #                     linecolor=color_palette[profilecolor(profile_type)],
            #                     markeralpha=markeralpha*0.4,
            #                     linealpha=markeralpha*0.5)
            # end

            
        end

        return nothing
    end

    """
        plot_bivariate_profiles_comparison(model::LikelihoodModel,
            xlim_scaler::Real=0.2,
            ylim_scaler::Real=0.2;
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
            compare_within_methods::Bool=false,
            include_dim_samples::Bool=false,
            palette_to_use::Symbol=:Paired_6, 
            markeralpha::Number=0.7,
            label_only_MLE::Bool=false,
            kwargs...)

    Returns a vector of comparison plots of bivariate profiles contained with the `model` struct that meet the requirements of the bivariate method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). Comparisons are between `profile_types` at the same `confidence_level` and `dof` for a given parameter combination; will also be within methods if `compare_within_methods=true`.

    The profiles plotted are based on the specified `θcombinations_to_plot`, `confidence_levels`, `dofs`, `profile_types`, `methods` and `sample_types`. By default, will plot all bivariate profiles generated. If `include_dim_samples=true` it will also include the concave hull boundary of bivariate profiles generated by an [`AbstractSampleType`](@ref) in the comparison (using [`LikelihoodBasedProfileWiseAnalysis.bivariate_concave_hull`](@ref)).

    If `label_only_MLE=true`, then only the MLE point will be labelled in the legend. Otherwise, profiles will be labelled by their `profile_type` or `sample_type`.

    `xlim_scaler` and `ylim_scaler` are used to uniformly push the `xlimits` and `ylimits` away from the location of the confidence boundary - if they are zero, then the extrema of the confidence boundary gives the location of the `xlimits` and the `ylimits`. If they are `1` then the corresponding limits have a range 100% wider than the extrema of the confidence boundary.
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_bivariate_profiles_comparison(model::LikelihoodModel,
                                        xlim_scaler::Real=0.2,
                                        ylim_scaler::Real=0.2;
                                        θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                        confidence_levels::Vector{<:Float64}=Float64[],
                                        dofs::Vector{<:Int}=Int[],
                                        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                        sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                        compare_within_methods::Bool=false,
                                        include_dim_samples::Bool=false,
                                        palette_to_use::Symbol=:Paired_6, 
                                        markeralpha::Number=0.7,
                                        label_only_MLE::Bool=false,
                                        kwargs...)

        θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)

        sub_df =LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, θcombinations_to_plot, confidence_levels,
                                    dofs, profile_types, methods)

        sub_df_samples = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_levels, dofs, sample_types, 
                                            sample_dimension=2)

        if ((!include_dim_samples || compare_within_methods) && nrow(sub_df) < 1) || 
            (nrow(sub_df) < 1 && nrow(sub_df_samples) < 1)
            return nothing
        end

        if isempty(confidence_levels)
            confidence_levels = unique(sub_df.conf_level)
        end
        if isempty(dofs)
            dofs = unique(sub_df.dof)
        end

        if compare_within_methods
            if isempty(methods); methods = unique(sub_df.method) end
            if isempty(confidence_levels); confidence_levels = unique(sub_df.conf_level) end
            
            θcombinations = [[inds...] for inds in unique(sub_df.θindices)]
        else
            if include_dim_samples
                if isempty(confidence_levels); confidence_levels = unique(vcat(sub_df.conf_level, sub_df_samples.conf_level)) end
                if isempty(sample_types) && nrow(sub_df_samples) > 0; sample_types = unique(sub_df_samples.sample_type) end

                θcombinations = unique(vcat([[inds...] for inds in unique(sub_df.θindices)],  sub_df_samples.θindices))
            else
                if isempty(confidence_levels); confidence_levels = unique(sub_df.conf_level) end

                θcombinations = [[inds...] for inds in unique(sub_df.θindices)]
            end
            
            if isempty(profile_types) && nrow(sub_df) > 0; profile_types = unique(sub_df.profile_type) end
        end

        color_palette = palette(palette_to_use)
        profile_plots = [plot() for _ in 1:nrow(sub_df)]

        profile_plots = [plot()]
        plot_i=1

        row_subset = trues(nrow(sub_df))
        row_subset_samples = trues(nrow(sub_df_samples))

        for θindices in θcombinations
            θindices_tuple = tuple(θindices...)
            parMLEs = model.core.θmle[θindices]
            θnames = model.core.θnames[θindices]

            for dof in dofs
                for confidence_level in confidence_levels
                    if compare_within_methods
                        for method in methods
                            row_subset .= (sub_df.θindices .== Ref(θindices_tuple)) .& 
                                            (sub_df.dof .== dof) .&
                                            (sub_df.conf_level .== confidence_level) .&
                                            (sub_df.method .== Ref(method))

                            conf_df = @view(sub_df[row_subset, :])

                            if nrow(conf_df) < 2; 
                                continue 
                            end

                            if plot_i > 1
                                append!(profile_plots, [plot()])
                            end
                            
                            min_vals = zeros(2)
                            max_vals = zeros(2)

                            for i in 1:nrow(conf_df)
                                row = @view(conf_df[i,:])

                                full_boundary = model.biv_profiles_dict[row.row_ind].confidence_boundary
                                boundary = @view(full_boundary[θindices, :])

                                if i == 1
                                    min_vals .= minimum(boundary, dims=2)
                                    max_vals .= maximum(boundary, dims=2)
                                else
                                    min_vals .= min.(min_vals, minimum(boundary, dims=2))
                                    max_vals .= max.(max_vals, maximum(boundary, dims=2))
                                end

                                plot2Dboundary!(profile_plots[plot_i], boundary, 
                                    label=label_only_MLE ? "" : string(row.profile_type),
                                    markershape=profile2Dmarkershape(row.profile_type, true), 
                                    markercolor=color_palette[profilecolor(row.profile_type)],
                                    linecolor=color_palette[profilecolor(row.profile_type)],
                                    markeralpha=markeralpha,
                                    linealpha=markeralpha)
                            end

                            ranges = max_vals .- min_vals

                            addMLE!(profile_plots[plot_i], parMLEs; 
                                markercolor=color_palette[end],
                                xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                                xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                                        max_vals[1]+ranges[1]*xlim_scaler],
                                ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                                max_vals[2]+ranges[2]*ylim_scaler],
                                title=string("Method: ", method, 
                                            "\nConfidence level: ", confidence_level, ", dof: ", dof),
                                titlefontsize=10, kwargs...)

                            plot_i+=1
                        end
                    else
                        row_subset .= (sub_df.θindices .== Ref(θindices_tuple)) .& 
                                            (sub_df.dof .== dof) .& 
                                            (sub_df.conf_level .== confidence_level)

                        conf_df = @view(sub_df[row_subset, :])
                        nrow_conf_df = nrow(conf_df)

                        if include_dim_samples
                            row_subset_samples .= (sub_df_samples.θindices .== Ref(θindices)) .& 
                                                (sub_df_samples.dof .== dof) .&
                                                (sub_df_samples.conf_level .== confidence_level)
                            conf_df_samples = @view(sub_df_samples[row_subset_samples, :])
                            nrow_conf_df_samples = nrow(conf_df_samples)
                        else
                            conf_df_samples=nothing
                            nrow_conf_df_samples=0
                        end

                        if !( nrow_conf_df > 1 && length(unique(conf_df.profile_type)) > 1 || nrow_conf_df_samples > 1 || (nrow_conf_df_samples == 1 && nrow_conf_df > 0) 
                            )
                            continue
                        end

                        if plot_i > 1
                            append!(profile_plots, [plot()])
                        end
                        
                        min_vals = zeros(2)
                        max_vals = zeros(2)
                        i = 1
                        for profile_type in profile_types
                            rows = @view(conf_df[conf_df.profile_type .== Ref(profile_type),:])
                            boundary = zeros(2,0)

                            if nrow(rows) == 0
                                continue
                            elseif nrow(rows) == 1
                                full_boundary = model.biv_profiles_dict[rows.row_ind[1]].confidence_boundary
                                boundary = @view(full_boundary[θindices, :])
                            else
                                for j in 1:nrow(rows)
                                    full_boundary = model.biv_profiles_dict[rows.row_ind[j]].confidence_boundary
                                    
                                    boundary = reduce(hcat, (boundary, full_boundary[θindices, :]))
                                end
                            end


                            if i == 1
                                min_vals .= minimum(boundary, dims=2)
                                max_vals .= maximum(boundary, dims=2)
                            else
                                min_vals .= min.(min_vals, minimum(boundary, dims=2))
                                max_vals .= max.(max_vals, maximum(boundary, dims=2))
                            end

                            plot2Dboundary!(profile_plots[plot_i], boundary,
                                label=label_only_MLE ? "" : string(profile_type),
                                markershape=profile2Dmarkershape(profile_type, true), 
                                markercolor=color_palette[profilecolor(profile_type)],
                                linecolor=color_palette[profilecolor(profile_type)],
                                markeralpha=markeralpha,
                                linealpha=markeralpha)
                            
                            i += 1
                        end

                        if !isnothing(conf_df_samples)
                            for sample_type in sample_types
                                row = @view(conf_df_samples[conf_df_samples.sample_type .== Ref(sample_type),:])
                                boundary = zeros(2,0)

                                if nrow(row) == 0
                                    continue
                                else
                                    boundary = LikelihoodBasedProfileWiseAnalysis.bivariate_concave_hull(model.dim_samples_dict[row.row_ind[1]], θindices, 1.0, 1.0, 
                                                                    LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), dof), sample_type)
                                end

                                if i == 1
                                    min_vals .= minimum(boundary, dims=2)
                                    max_vals .= maximum(boundary, dims=2)
                                else
                                    min_vals .= min.(min_vals, minimum(boundary, dims=2))
                                    max_vals .= max.(max_vals, maximum(boundary, dims=2))
                                end

                                plot2Dboundary!(profile_plots[plot_i], boundary, 
                                    label=string(sample_type),
                                    markershape=profile2Dmarkershape(sample_type, true), 
                                    markercolor=color_palette[profilecolor(sample_type)],
                                    linecolor=color_palette[profilecolor(sample_type)],
                                    markeralpha=markeralpha,
                                    linealpha=markeralpha)
                                
                                i += 1
                            end
                        end

                        ranges = max_vals .- min_vals

                        addMLE!(profile_plots[plot_i], parMLEs; 
                            markercolor=color_palette[end],
                            xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                            xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                                    max_vals[1]+ranges[1]*xlim_scaler],
                            ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                            max_vals[2]+ranges[2]*ylim_scaler],
                            title=string("Confidence level: ", confidence_level, ", dof: ", dof),
                            titlefontsize=10, kwargs...)

                        plot_i+=1
                    end
                end
            end
        end

        return profile_plots
    end

    """
        plot_predictions_individual(model::LikelihoodModel,
            t::AbstractVector,
            profile_dimension::Int=1;
            xlabel::String="t",
            ylabel::Union{Nothing,String,Vector{String}}=nothing,
            for_dim_samples::Bool=false,
            include_MLE::Bool=true,
            θs_to_plot::Vector=Int[],
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            θindices_to_plot::Vector=Vector{Int}[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
            linealpha=0.4, 
            kwargs...)

    Returns a vector of plots of profile-wise predictions of the model trajectory formed from profiles with interest parameter dimension `profile_dimension` that meet the requirement of the relevant method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). 

    The plotted extrema are the extrema of approximate profile-wise `confidence_level` trajectory confidence set from each profile.

    `t` should be the same points used to generate predictions in [`generate_predictions_univariate!`](@ref), [`generate_predictions_bivariate!`](@ref) and [`generate_predictions_dim_samples!`](@ref). 

    The profiles plotted are based on the specified `θs_to_plot`, `θcombinations_to_plot`, `θs_to_plot`, `confidence_levels`, `dofs`, `profile_types`, `methods` and `sample_types`. By default, will plot all predictions generated from profiles with `profile_dimension`. If `for_dim_samples=true` then profile-wise trajectory confidence sets will be plotted from profiles sampled using an [`AbstractSampleType`](@ref).

    `linealpha` is the alpha value used for plotting each individual model trajectory line contained within a profile-wise trajectory confidence set. 
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_predictions_individual(model::LikelihoodModel,
                                # prediction_type::Symbol=:union,
                                t::AbstractVector,
                                profile_dimension::Int=1;
                                xlabel::String="t",
                                ylabel::Union{Nothing,String,Vector{String}}=nothing,
                                for_dim_samples::Bool=false,
                                # ylim_scaler::Real=0.2;
                                include_MLE::Bool=true,
                                θs_to_plot::Vector=Int[],
                                θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                θindices_to_plot::Vector=Vector{Int}[],
                                confidence_levels::Vector{<:Float64}=Float64[],
                                dofs::Vector{<:Int}=Int[],
                                profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                                methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                # palette_to_use::Symbol=:Paired_6, 
                                linealpha=0.4, 
                                kwargs...)
                                
        if for_dim_samples
            if !(1 ≤ profile_dimension && profile_dimension ≤ model.core.num_pars)
                throw(DomainError(string("profile_dimension must be between 1 and ", 
                        model.core.num_pars, " (the number of model parameters)")))
            end

            θindices_to_plot = θindices_to_plot isa Vector{Vector{Symbol}} ? 
                                LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices(model, θindices_to_plot) :
                                θindices_to_plot
            sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_levels, dofs, sample_types;
                                        sample_dimension=profile_dimension, for_prediction_plots=true)
            predictions_dict = model.dim_predictions_dict
        else
            profile_dimension in [1,2] || throw(DomainError("profile_dimension must be 1 or 2"))

            if profile_dimension == 1
                θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, θs_to_plot, confidence_levels,
                                            dofs, profile_types; for_prediction_plots=true)
                predictions_dict = model.uni_predictions_dict
            elseif profile_dimension == 2
                θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, θcombinations_to_plot, confidence_levels,
                                            dofs, profile_types, methods; for_prediction_plots=true)
                predictions_dict = model.biv_predictions_dict
            end
        end

        if nrow(sub_df) < 1
            return nothing
        end
        
        # color_palette = palette(palette_to_use)
        layout = ndims(model.core.ymle) > 1 ? size(model.core.ymle,2) : 1
        prediction_plots = [plot(layout=layout) for _ in 1:nrow(sub_df)]

        for i in 1:nrow(sub_df)

            row = @view(sub_df[i,:])
            predictions = predictions_dict[row.row_ind].predictions
            extrema = predictions_dict[row.row_ind].extrema

            if for_dim_samples
                title=string("Sample type: ", row.sample_type, 
                            "\nConfidence level: ", row.conf_level, ", dof: ", row.dof,
                            "\nTarget parameter(s): ", model.core.θnames[row.θindices])
                title_vspan = 0.15
            else
                if profile_dimension == 1
                    title=string("Profile type: ", row.profile_type, 
                                "\nConfidence level: ", row.conf_level, ", dof: ", row.dof,
                                "\nTarget parameter: ", model.core.θnames[row.θindex])
                    title_vspan = 0.15
                else
                    θindices = zeros(Int,2); 
                    for j in 1:2; θindices[j] = row.θindices[j] end

                    title=string("Profile type: ", row.profile_type, 
                                "\nMethod: ", row.method, 
                                "\nConfidence level: ", row.conf_level, ", dof: ", row.dof,
                                "\nTarget parameters: ", model.core.θnames[θindices])
                    title_vspan = 0.2
                end
            end
            
            # y_extrema = [minimum(extrema[:,1]), maximum(extrema[:,2])]
            # range = diff(y_extrema)[1]

            # ylims=[y_extrema[1]-range*ylim_scaler, y_extrema[2]+range*ylim_scaler]
            title_args = layout==1 ? (title=title, titlefontsize=10, top_margin=(5,:mm)) : (plot_title=title, plot_titlefontsize=10, plot_titlevspan=0.2)

            plotprediction!(prediction_plots[i], t, predictions, extrema, linealpha, layout; 
                            xlabel=xlabel,
                            # ylims=ylims,
                            title_args...,
                            kwargs...)

            if layout > 1
                if isnothing(ylabel); ylabel =[string("f", j, "(", xlabel, ")") for j in 1:layout] end

                if ylabel isa String
                    ylabel!(prediction_plots[i], ylabel)
                else
                    for j in 1:layout; ylabel!(prediction_plots[i][j], ylabel[j]) end
                end
            else
                if isnothing(ylabel); ylabel = string("f(", xlabel, ")") end
                ylabel!(prediction_plots[i], ylabel)
            end
        end

        if include_MLE 
            ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
            for plt in prediction_plots
               LikelihoodBasedProfileWiseAnalysis. add_yMLE!(plt, t, ymle, layout)
            end
        end

        return prediction_plots
    end

    """
        plot_predictions_union(model::LikelihoodModel,
            t::AbstractVector,
            profile_dimension::Int=1,
            confidence_level::Float64=0.95;
            dof::Int=profile_dimension,
            xlabel::String="t",
            ylabel::Union{Nothing,String,Vector{String}}=nothing,
            for_dim_samples::Bool=false,
            include_MLE::Bool=true,
            θs_to_plot::Vector = Int[],
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            θindices_to_plot::Vector=Vector{Int}[],
            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
            compare_to_full_sample_type::Union{Missing, AbstractSampleType}=missing,
            include_lower_confidence_levels::Bool=false,
            linealpha=0.4,
            kwargs...)


    Returns a plot of the union of profile-wise predictions of the model trajectory formed from profiles with interest parameter dimension `profile_dimension` that meet the requirement of the relevant method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). 

    The plotted extrema are the extrema of the approximate profile-wise `confidence_level` trajectory confidence set. 

    `t` should be the same points used to generate predictions in [`generate_predictions_univariate!`](@ref), [`generate_predictions_bivariate!`](@ref) and [`generate_predictions_dim_samples!`](@ref). 

    The profiles plotted are based on the specified `θs_to_plot`, `θcombinations_to_plot`, `θs_to_plot`, `confidence_levels`, `dofs`, `profile_types`, `methods` and `sample_types`. By default, will plot all predictions generated from profiles with `profile_dimension`. If `for_dim_samples=true` then the profile-wise trajectory confidence set will be plotted from profiles sampled using an [`AbstractSampleType`](@ref).

    `include_lower_confidence_levels` is only relevant for profiles of dimension 2 evaluated using an [`AbstractBivariateMethod`](@ref).

    If `compare_to_full_sample_type isa AbstractSampleType` then will also plot the extrema of the trajectory confidence set from a full parameter confidence set evaluated using the specified [`AbstractSampleType`](@ref). For example use `compare_to_full_sample_type=LatinHypercubeSamples()`.

    `linealpha` is the alpha value used for plotting each individual model trajectory line contained within a profile-wise trajectory confidence set. 
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_predictions_union(model::LikelihoodModel,
                                t::AbstractVector,
                                profile_dimension::Int=1,
                                confidence_level::Float64=0.95;
                                dof::Int=profile_dimension,
                                xlabel::String="t",
                                ylabel::Union{Nothing,String,Vector{String}}=nothing,
                                for_dim_samples::Bool=false,
                                include_MLE::Bool=true,
                                θs_to_plot::Vector = Int[],
                                θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                θindices_to_plot::Vector=Vector{Int}[],
                                profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                                methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                # palette_to_use::Symbol=:Paired_6, 
                                # union_within_methods::Bool=false,
                                compare_to_full_sample_type::Union{Missing, AbstractSampleType}=missing,
                                include_lower_confidence_levels::Bool=false,
                                linealpha=0.4,
                                kwargs...)

        if for_dim_samples
            if !(1 ≤ profile_dimension && profile_dimension ≤ model.core.num_pars)
                    throw(DomainError(string("profile_dimension must be between 1 and ", 
                        model.core.num_pars, " (the number of model parameters)")))
            end

            θindices_to_plot = θindices_to_plot isa Vector{Vector{Symbol}} ? 
                                LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices(model, θindices_to_plot) :
                                θindices_to_plot
            sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_level, dof, sample_types; 
                                        sample_dimension=profile_dimension, for_prediction_plots=true)
            predictions_dict = model.dim_predictions_dict
            title = string("Parameter dimension: " , profile_dimension,
                            "\nMethod: sampled",
                            "\nConfidence level: ", confidence_level, ", dof: ", dof)
        else
            profile_dimension in [1,2] || throw(DomainError("profile_dimension must be 1 or 2"))
            if profile_dimension == 1
                θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.uni_profiles_df, 
                    model.num_uni_profiles, θs_to_plot, confidence_level, dof, profile_types; for_prediction_plots=true)
                predictions_dict = model.uni_predictions_dict
            elseif profile_dimension == 2
                θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, 
                    model.num_biv_profiles, θcombinations_to_plot, confidence_level, dof, profile_types, methods; for_prediction_plots=true, include_lower_confidence_levels)
                predictions_dict = model.biv_predictions_dict
            end

            title = string("Parameter dimension: " , profile_dimension,
                            "\nMethod: boundary",
                            "\nConfidence level: ", confidence_level, ", dof: ", dof)
        end

        if nrow(sub_df) < 1
            return nothing
        end

        layout = ndims(model.core.ymle) > 1 ? size(model.core.ymle,2) : 1
        prediction_plot = plot(layout=layout)

        predictions_union = []
        extrema_union = []

        for i in 1:nrow(sub_df)
            row = @view(sub_df[i,:])
            predictions = predictions_dict[row.row_ind].predictions
            extrema = predictions_dict[row.row_ind].extrema
            
            if i == 1
                predictions_union = predictions .* 1.0
                extrema_union = extrema .* 1.0
            else
                predictions_union = reduce(hcat, (predictions_union, predictions))
                if layout > 1
                    extrema_union[:,1,:] .= min.(extrema_union[:,1,:], extrema[:,1,:])
                    extrema_union[:,2,:] .= max.(extrema_union[:,2,:], extrema[:,2,:])
                else
                    extrema_union[:,1] .= min.(extrema_union[:,1], extrema[:,1])
                    extrema_union[:,2] .= max.(extrema_union[:,2], extrema[:,2])
                end
            end
        end

        title_args = layout==1 ? (title=title, titlefontsize=10, top_margin=(5,:mm)) : (plot_title=title, plot_titlefontsize=10, plot_titlevspan=0.15)

        plotprediction!(prediction_plot, t, predictions_union, extrema_union, linealpha, layout; 
                        xlabel=xlabel,
                        title_args...,
                        kwargs...)

        if layout > 1
            if isnothing(ylabel); ylabel =[string("f", j, "(", xlabel, ")") for j in 1:layout] end

            if ylabel isa String
                ylabel!(prediction_plots[i], ylabel)
            else
                for j in 1:layout; ylabel!(prediction_plot[j], ylabel[j]) end
            end
        else
            if isnothing(ylabel); ylabel = string("f(", xlabel, ")") end
            ylabel!(prediction_plot, ylabel)
        end

        if !ismissing(compare_to_full_sample_type)
            row_subset = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_level, model.core.num_pars,
                                            [compare_to_full_sample_type], sample_dimension=model.core.num_pars, 
                                            for_prediction_plots=true,
                                            include_higher_confidence_levels=true)
            
            if nrow(row_subset) > 0
                if nrow(row_subset) > 1
                    subset_to_use = @view(row_subset[argmin(row_subset.conf_level), :])
                elseif nrow(row_subset) == 1
                    subset_to_use = @view(row_subset[1,:])
                end

                if subset_to_use.conf_level == confidence_level
                    sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].extrema
                else
                    ll_level = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(model, confidence_level, 
                                                        EllipseApproxAnalytical(), model.core.num_pars)

                    valid_point = model.dim_samples_dict[subset_to_use.row_ind].ll .> ll_level 

                    if layout > 1
                        sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].extrema[valid_point,:,:]
                    else
                        sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].extrema[valid_point,:]
                    end
                end

                add_extrema!(prediction_plot, t, sampled_extrema, layout)
            end
        end

        if include_MLE 
            ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
            add_yMLE!(prediction_plot, t, ymle, layout)
        end

        return prediction_plot
    end

    """
        plot_realisations_individual(model::LikelihoodModel,
            t::AbstractVector,
            profile_dimension::Int=1;
            xlabel::String="t",
            ylabel::Union{Nothing,String,Vector{String}}=nothing,
            for_dim_samples::Bool=false
            include_MLE::Bool=true,
            θs_to_plot::Vector=Int[],
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            θindices_to_plot::Vector=Vector{Int}[],
            confidence_levels::Vector{<:Float64}=Float64[],
            dofs::Vector{<:Int}=Int[],
            regions::Vector{<:Real}=Float64[],
            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
            linealpha=0.4, 
            kwargs...)

    Returns a vector of plots of profile-wise predictions of the `region` population reference set formed from profiles with interest parameter dimension `profile_dimension` that meet the requirement of the relevant method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). 

    The plotted extrema are the extrema of the approximate profile-wise (`region`, `confidence_level`) reference tolerance set from each profile. 

    `t` should be the same points used to generate predictions in [`generate_predictions_univariate!`](@ref), [`generate_predictions_bivariate!`](@ref) and [`generate_predictions_dim_samples!`](@ref). 

    The profiles plotted are based on the specified `θs_to_plot`, `θcombinations_to_plot`, `θs_to_plot`, `confidence_levels`, `dofs`, `regions`, `profile_types`, `methods` and `sample_types`. By default, will plot all predictions generated from profiles with `profile_dimension`. If `for_dim_samples=true` then profile-wise trajectory confidence sets will be plotted from profiles sampled using an [`AbstractSampleType`](@ref).
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_realisations_individual(model::LikelihoodModel,
                                # prediction_type::Symbol=:union,
                                t::AbstractVector,
                                profile_dimension::Int=1;
                                xlabel::String="t",
                                ylabel::Union{Nothing,String,Vector{String}}=nothing,
                                for_dim_samples::Bool=false,
                                # ylim_scaler::Real=0.2;
                                include_MLE::Bool=true,
                                θs_to_plot::Vector=Int[],
                                θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                θindices_to_plot::Vector=Vector{Int}[],
                                confidence_levels::Vector{<:Float64}=Float64[],
                                dofs::Vector{<:Int}=Int[],
                                regions::Vector{<:Real}=Float64[],
                                profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                                methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                # palette_to_use::Symbol=:Paired_6, 
                                linealpha=0.4, 
                                kwargs...)
                                
        if for_dim_samples
            if !(1 ≤ profile_dimension && profile_dimension ≤ model.core.num_pars)
                throw(DomainError(string("profile_dimension must be between 1 and ", 
                        model.core.num_pars, " (the number of model parameters)")))
            end

            θindices_to_plot = θindices_to_plot isa Vector{Vector{Symbol}} ? 
                                LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices(model, θindices_to_plot) :
                                θindices_to_plot
            sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_levels, dofs, sample_types;
                                        sample_dimension=profile_dimension, regions=regions, for_prediction_plots=true)
            predictions_dict = model.dim_predictions_dict
        else
            profile_dimension in [1,2] || throw(DomainError("profile_dimension must be 1 or 2"))

            if profile_dimension == 1
                θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, θs_to_plot, confidence_levels,
                                            dofs, profile_types; regions=regions, for_prediction_plots=true)
                predictions_dict = model.uni_predictions_dict
            elseif profile_dimension == 2
                θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, θcombinations_to_plot, confidence_levels,
                                            dofs, profile_types, methods; regions=regions, for_prediction_plots=true)
                predictions_dict = model.biv_predictions_dict
            end
        end

        if nrow(sub_df) < 1
            return nothing
        end
        
        # color_palette = palette(palette_to_use)
        layout = ndims(model.core.ymle) > 1 ? size(model.core.ymle,2) : 1
        realisation_plots = [plot(layout=layout) for _ in 1:nrow(sub_df)]

        for i in 1:nrow(sub_df)

            row = @view(sub_df[i,:])

            realisations = predictions_dict[row.row_ind].realisations
            if isempty(realisations.lq)
                continue
            end
            extrema = realisations.extrema

            if for_dim_samples
                title=string("Sample type: ", row.sample_type, 
                            "\nConfidence level: ", row.conf_level, ", dof: ", row.dof,
                            "\nReference region: ", row.region,
                            "\nTarget parameter(s): ", model.core.θnames[row.θindices])
                title_vspan = 0.15
            else
                if profile_dimension == 1
                    title=string("Profile type: ", row.profile_type, 
                                "\nConfidence level: ", row.conf_level, ", dof: ", row.dof,
                                "\nReference region: ", row.region,
                                "\nTarget parameter: ", model.core.θnames[row.θindex])
                    title_vspan = 0.15
                else
                    θindices = zeros(Int,2); 
                    for j in 1:2; θindices[j] = row.θindices[j] end

                    title=string("Profile type: ", row.profile_type, 
                                "\nMethod: ", row.method, 
                                "\nConfidence level: ", row.conf_level, ", dof: ", row.dof,
                                "\nReference region: ", row.region,
                                "\nTarget parameters: ", model.core.θnames[θindices])
                    title_vspan = 0.2
                end
            end
            
            # y_extrema = [minimum(extrema[:,1]), maximum(extrema[:,2])]
            # range = diff(y_extrema)[1]

            # ylims=[y_extrema[1]-range*ylim_scaler, y_extrema[2]+range*ylim_scaler]
            title_args = layout==1 ? (title=title, titlefontsize=10, top_margin=(5,:mm)) : (plot_title=title, plot_titlefontsize=10, plot_titlevspan=0.2)

            plotrealisation!(realisation_plots[i], t, extrema, linealpha, layout; 
                            xlabel=xlabel,
                            # ylims=ylims,
                            title_args...,
                            kwargs...)

            if layout > 1
                if isnothing(ylabel); ylabel =[string("f", j, "(", xlabel, ")") for j in 1:layout] end

                if ylabel isa String
                    ylabel!(realisation_plots[i], ylabel)
                else
                    for j in 1:layout; ylabel!(realisation_plots[i][j], ylabel[j]) end
                end
            else
                if isnothing(ylabel); ylabel = string("f(", xlabel, ")") end
                ylabel!(realisation_plots[i], ylabel)
            end
        end

        if include_MLE 
            ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
            for plt in realisation_plots
                add_yMLE!(plt, t, ymle, layout)
            end
        end

        return realisation_plots
    end

    """
        plot_realisations_union(model::LikelihoodModel,
            t::AbstractVector,
            profile_dimension::Int=1,
            confidence_level::Float64=0.95;
            dof::Int=profile_dimension,
            region::Real=0.95,
            xlabel::String="t",
            ylabel::Union{Nothing,String,Vector{String}}=nothing,
            for_dim_samples::Bool=false,
            include_MLE::Bool=true,
            θs_to_plot::Vector = Int[],
            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
            θindices_to_plot::Vector=Vector{Int}[],
            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
            compare_to_full_sample_type::Union{Missing, AbstractSampleType}=missing,
            include_lower_confidence_levels::Bool=false,
            linealpha=0.4,
            kwargs...)

    Returns a plot of the union of profile-wise predictions of the `region` population reference set formed from profiles with interest parameter dimension `profile_dimension` that meet the requirement of the relevant method of [`LikelihoodBasedProfileWiseAnalysis.desired_df_subset`](@ref) (see Keyword Arguments). 

    The plotted extrema are the extrema of the approximate profile-wise (`region`, `confidence_level`) reference tolerance set. 

    `t` should be the same points used to generate predictions in [`generate_predictions_univariate!`](@ref), [`generate_predictions_bivariate!`](@ref) and [`generate_predictions_dim_samples!`](@ref). 

    The profiles plotted are based on the specified `θs_to_plot`, `θcombinations_to_plot`, `θs_to_plot`, `confidence_levels`, `dofs`, `regions`, `profile_types`, `methods` and `sample_types`. By default, will plot all predictions generated from profiles with `profile_dimension`. If `for_dim_samples=true` then the profile-wise reference tolerancce set will be plotted from profiles sampled using an [`AbstractSampleType`](@ref).

    `include_lower_confidence_levels` is only relevant for profiles of dimension 2 evaluated using an [`AbstractBivariateMethod`](@ref).

    If `compare_to_full_sample_type isa AbstractSampleType` then will also plot the extrema of the reference tolerance set from a full parameter confidence set evaluated using the specified [`AbstractSampleType`](@ref). For example use `compare_to_full_sample_type=LatinHypercubeSamples()`.
    """
    function LikelihoodBasedProfileWiseAnalysis.plot_realisations_union(model::LikelihoodModel,
                                t::AbstractVector,
                                profile_dimension::Int=1,
                                confidence_level::Float64=0.95;
                                dof::Int=profile_dimension,
                                region::Real=0.95,
                                xlabel::String="t",
                                ylabel::Union{Nothing,String,Vector{String}}=nothing,
                                for_dim_samples::Bool=false,
                                include_MLE::Bool=true,
                                θs_to_plot::Vector = Int[],
                                θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                θindices_to_plot::Vector=Vector{Int}[],
                                profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                                methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                # palette_to_use::Symbol=:Paired_6, 
                                # union_within_methods::Bool=false,
                                compare_to_full_sample_type::Union{Missing, AbstractSampleType}=missing,
                                include_lower_confidence_levels::Bool=false,
                                linealpha=0.4,
                                kwargs...)

        if for_dim_samples
            if !(1 ≤ profile_dimension && profile_dimension ≤ model.core.num_pars)
                    throw(DomainError(string("profile_dimension must be between 1 and ", 
                        model.core.num_pars, " (the number of model parameters)")))
            end

            θindices_to_plot = θindices_to_plot isa Vector{Vector{Symbol}} ? 
                                LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices(model, θindices_to_plot) :
                                θindices_to_plot
            sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_level, dof, sample_types; 
                                        sample_dimension=profile_dimension, regions=region, for_prediction_plots=true)
            predictions_dict = model.dim_predictions_dict
            title = string("Parameter dimension: " , profile_dimension,
                            "\nMethod: sampled",
                            "\nConfidence level: ", confidence_level, ", dof: ", dof,
                            "\nReference region: ", region)
        else
            profile_dimension in [1,2] || throw(DomainError("profile_dimension must be 1 or 2"))
            if profile_dimension == 1
                θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.uni_profiles_df, 
                    model.num_uni_profiles, θs_to_plot, confidence_level, 
                    dof, profile_types; regions=region, for_prediction_plots=true)
                predictions_dict = model.uni_predictions_dict
            elseif profile_dimension == 2
                θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
                sub_df = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.biv_profiles_df, 
                    model.num_biv_profiles, θcombinations_to_plot, confidence_level, 
                    dof, profile_types, methods; regions=region, for_prediction_plots=true, include_lower_confidence_levels)
                predictions_dict = model.biv_predictions_dict
            end

            title = string("Parameter dimension: " , profile_dimension,
                            "\nMethod: boundary",
                            "\nConfidence level: ", confidence_level, ", dof: ", dof,
                            "\nReference region: ", region)
        end

        if nrow(sub_df) < 1
            return nothing
        end

        layout = ndims(model.core.ymle) > 1 ? size(model.core.ymle,2) : 1
        realisation_plot = plot(layout=layout)

        extrema_union = []

        j=0
        for i in 1:nrow(sub_df)
            row = @view(sub_df[i,:])

            realisations = predictions_dict[row.row_ind].realisations
            if isempty(realisations.lq)
                continue
            end
            extrema = realisations.extrema
            j+=1
            
            if j == 1
                extrema_union = extrema .* 1.0
            else
                if layout > 1
                    extrema_union[:,1,:] .= min.(extrema_union[:,1,:], extrema[:,1,:])
                    extrema_union[:,2,:] .= max.(extrema_union[:,2,:], extrema[:,2,:])
                else
                    extrema_union[:,1] .= min.(extrema_union[:,1], extrema[:,1])
                    extrema_union[:,2] .= max.(extrema_union[:,2], extrema[:,2])
                end
            end
        end

        title_args = layout==1 ? (title=title, titlefontsize=10, top_margin=(5,:mm)) : (plot_title=title, plot_titlefontsize=10, plot_titlevspan=0.15)

        plotrealisation!(realisation_plot, t, extrema_union, linealpha, layout; 
                        xlabel=xlabel,
                        title_args...,
                        kwargs...)

        if layout > 1
            if isnothing(ylabel); ylabel =[string("f", j, "(", xlabel, ")") for j in 1:layout] end

            if ylabel isa String
                ylabel!(realisation_plots[i], ylabel)
            else
                for j in 1:layout; ylabel!(realisation_plot[j], ylabel[j]) end
            end
        else
            if isnothing(ylabel); ylabel = string("f(", xlabel, ")") end
            ylabel!(realisation_plot, ylabel)
        end

        if !ismissing(compare_to_full_sample_type)
            row_subset = LikelihoodBasedProfileWiseAnalysis.desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_level, model.core.num_pars,
                                            [compare_to_full_sample_type], sample_dimension=model.core.num_pars, 
                                            regions=region,
                                            for_prediction_plots=true,
                                            include_higher_confidence_levels=true)
            
            if nrow(row_subset) > 0
                if nrow(row_subset) > 1
                    subset_to_use = @view(row_subset[argmin(row_subset.conf_level), :])
                elseif nrow(row_subset) == 1
                    subset_to_use = @view(row_subset[1,:])
                end

                if subset_to_use.conf_level == confidence_level
                    sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].realisations.extrema
                else
                    ll_level = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(model, confidence_level, 
                                                        EllipseApproxAnalytical(), model.core.num_pars)

                    valid_point = model.dim_samples_dict[subset_to_use.row_ind].ll .> ll_level 

                    if layout > 1
                        sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].realisations.extrema[valid_point,:,:]
                    else
                        sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].realisations.extrema[valid_point,:]
                    end
                end

                add_extrema!(realisation_plot, t, sampled_extrema, layout; label=["Sampled SRTBs (≈)" ""])
            end
        end

        if include_MLE 
            ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
            add_yMLE!(realisation_plot, t, ymle, layout)
        end

        return realisation_plot
    end
end