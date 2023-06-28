function findNpointpairs_fix1axis!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    i::Int, 
                                    j::Int,
                                    mle_targetll::Float64,
                                    save_internal_points::Bool,
                                    biv_opt_is_ellipse_analytical::Bool,
                                    channel::RemoteChannel)

    x_vec, y_vec = zeros(num_points), zeros(2, num_points)
    local internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    local ll_values = zeros(save_internal_points ? num_points : 0)

    ψ_y0, ψ_y1 = 0.0, 0.0
    
    if biv_opt_is_ellipse_analytical

        for k in 1:num_points
            # do-while loop
            while true
                p.ψ_x[1] = rand(Uniform(model.core.θlb[i], model.core.θub[i]))
                ψ_y0 = rand(Uniform(model.core.θlb[j], model.core.θub[j]))
                ψ_y1 = rand(Uniform(model.core.θlb[j], model.core.θub[j])) 

                f0 = bivariate_optimiser(ψ_y0, p)
                f1 = bivariate_optimiser(ψ_y1, p)

                if f0 * f1 < 0
                    x_vec[k] = p.ψ_x[1]
                    y_vec[:,k] .= ψ_y0, ψ_y1

                    if save_internal_points
                        internal_all[i,k] = p.ψ_x[1]

                        if f0 ≥ 0 
                            ll_values[k] = f0
                            internal_all[j,k] = ψ_y0
                        else
                            ll_values[k] = f1
                            internal_all[j,k] = ψ_y1
                        end
                    end
                    break
                end
            end
            put!(channel, true)
        end

        if save_internal_points
            get_ωs_bivariate_ellipse_analytical!(@view(internal_all[[i, j], :]), num_points,
                                                    p.consistent, i, j, 
                                                    model.core.num_pars, p.initGuess,
                                                    p.θranges, p.ωranges, internal_all)
        end

    else    
        ω_opt0, ω_opt1 = zeros(model.core.num_pars-2), zeros(model.core.num_pars-2)

        for k in 1:num_points
            # do-while loop
            while true
                p.ψ_x[1] = rand(Uniform(model.core.θlb[i], model.core.θub[i]))
                ψ_y0 = rand(Uniform(model.core.θlb[j], model.core.θub[j]))
                ψ_y1 = rand(Uniform(model.core.θlb[j], model.core.θub[j])) 

                f0 = bivariate_optimiser(ψ_y0, p)
                ω_opt0 .= p.ω_opt
                f1 = bivariate_optimiser(ψ_y1, p)
                ω_opt1 .= p.ω_opt

                if f0 * f1 < 0
                    x_vec[k] = p.ψ_x[1]
                    y_vec[:,k] .= ψ_y0, ψ_y1

                    if save_internal_points
                        internal_all[i,k] = p.ψ_x[1]
                        if f0 ≥ 0 
                            ll_values[k] = f0
                            internal_all[j,k] = ψ_y0
                            variablemapping!(@view(internal_all[:, k]), ω_opt0, p.θranges, p.ωranges)
                        else
                            ll_values[k] = f1
                            internal_all[j,k] = ψ_y1
                            variablemapping!(@view(internal_all[:, k]), ω_opt1, p.θranges, p.ωranges)
                        end
                    end
                    break
                end
            end
            put!(channel, true)
        end
    end

    if save_internal_points; ll_values .= ll_values .+ mle_targetll end

    return x_vec, y_vec, internal_all, ll_values
end

function bivariate_confidenceprofile_fix1axis(bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int,
                                                mle_targetll::Float64,
                                                save_internal_points::Bool, 
                                                channel::RemoteChannel)

    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateψ_ellipse_analytical

    boundary = zeros(model.core.num_pars, num_points)
    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)

    count=0
    for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, (div(num_points,2) + rem(num_points,2))]]

        if biv_opt_is_ellipse_analytical
            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, ψ_x=[0.0],
                θranges=θranges, ωranges=ωranges, consistent=consistent)
        else
            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, ψ_x=[0.0],
                θranges=θranges, ωranges=ωranges, consistent=consistent, ω_opt=zeros(model.core.num_pars-2))
        end


        x_vec, y_vec, internal, ll = findNpointpairs_fix1axis!(p, bivariate_optimiser, model,
                                                            N, i, j, mle_targetll, save_internal_points,
                                                            biv_opt_is_ellipse_analytical, channel)
        
        for k in 1:N
            count +=1

            p.ψ_x[1] = x_vec[k]

            ψ_y1 = find_zero(bivariate_optimiser, (y_vec[1,k], y_vec[2,k]), Roots.Brent(); p=p)

            boundary[i, count] = x_vec[k]
            boundary[j, count] = ψ_y1
            
            if !biv_opt_is_ellipse_analytical
                bivariate_optimiser(ψ_y1, p)
                variablemapping!(@view(boundary[:, count]), p.ω_opt, θranges, ωranges)
            end
            put!(channel, true)
        end

        if save_internal_points
            internal_all = hcat(internal_all, internal)
            ll_values = vcat(ll_values, ll) 
        end
    end

    if biv_opt_is_ellipse_analytical
        return get_ωs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                                                    consistent, ind1, ind2, 
                                                    model.core.num_pars, initGuess,
                                                    θranges, ωranges, boundary), PointsAndLogLikelihood(internal_all, ll_values)
    end

    return boundary, PointsAndLogLikelihood(internal_all, ll_values)
end
