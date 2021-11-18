function perceptron(fit_data::Vector{<:Vector{<:Real}}, eta::Real)
    n = length(fit_data[1]); w = zeros(n-1); b = 0
    error = true
    while error
        error = false
        for data in fit_data
            if data[end]*(w'*data[1:end-1]+b) <= 0
                w += eta*data[end]*data[1:end-1]
                b += eta*data[end]
                error = true
            end
        end
    end
    return w, b
end

function perceptron_dual(fit_data::Vector{<:Vector{<:Real}}, eta::Real)
    error = true
    a = zeros(length(fit_data))
    b = 0.0
    while error 
        error = false
        w = sum([a[i]*fit_data[i][end]*fit_data[i][1:end-1] for i in 1:length(fit_data)])
        for (index, data) in enumerate(fit_data)
            if data[end]*(w'*data[1:end-1] + b) <= 0
                a[index] += eta
                b += eta*data[3]
                error = true
            end
        end 
    end
    return sum([a[i]*fit_data[i][1:end-1] for i in 1:length(fit_data)]), b               
end

function scat(fit_data::Vector{<:Vector{<:Real}})
    n = length(fit_data)
    if n==3
        scatter_x = [fit_data[i][1] for i in 1:n]
        scatter_y = [fit_data[i][2] for i in 1:n]
        scatter_color = [fit_data[i][end] for i in 1:n]
        scatter(scatter_x, scatter_y, color=scatter_color)
    elseif n==4
        scatter_x = [fit_data[i][1] for i in 1:n]
        scatter_y = [fit_data[i][2] for i in 1:n]
        scatter_z = [fit_data[i][3] for i in 1:n]
        scatter_color = [fit_data[i][end] for i in 1:n]
        scatter(scatter_x, scatter_y, scatter_z, color=scatter_color, markersize = 250)
    end
end

function scat!(ax::Union{Axis, LScene}, fit_data::Vector{<:Vector{<:Real}})
    n = length(fit_data)
    if n==3
        scatter_x = [fit_data[i][1] for i in 1:n]
        scatter_y = [fit_data[i][2] for i in 1:n]
        scatter_color = [fit_data[i][end] for i in 1:n]
        scatter!(ax, scatter_x, scatter_y, color=scatter_color)
    elseif n==4
        scatter_x = [fit_data[i][1] for i in 1:n]
        scatter_y = [fit_data[i][2] for i in 1:n]
        scatter_z = [fit_data[i][3] for i in 1:n]
        scatter_color = [fit_data[i][end] for i in 1:n]
        scatter!(ax, scatter_x, scatter_y, scatter_z, color=scatter_color, markersize = 250)
    end
end

function pplot(w::Vector{<:Real}, b::Real, range)
    f(x) = -(w[1]*x+b)/w[2]
    lines(range, f)
end

function pplot!(ax::Axis, w::Vector{<:Real}, b::Real, range)
    f(x) = -(w[1]*x+b)/w[2]
    lines!(ax, range, f)
end

function pplot(w::Vector{<:Real}, b::Real, x_range, y_range)
    f(x, y) = -(w[1]*x+w[2]*y+b)/w[3]
    surface(x_range, y_range, f)
end

function pplot!(ax::LScene, w::Vector{<:Real}, b::Real, x_range, y_range)
    f(x, y) = -(w[1]*x+w[2]*y+b)/w[3]
    surface!(ax, x_range, y_range, f)
end

mutable struct Kdtree
    location::Vector{<:Real}
    left::Union{Nothing, Kdtree}
    right::Union{Nothing, Kdtree}
end

function kdtree(input_data::Vector{<:Vector{<:Real}}; depth::Int64 = 1)    
    if length(input_data) == 0
        return nothing
    end
    data = reduce(hcat, input_data)
    dims = length(input_data[1])
    len = length(input_data)
    midn = ceil(Int64, len/2)
    index = sortperm(data[(depth+2) % dims + 1, :])
    left = index[1:midn-1]
    right = index[midn+1:len]
    Kdtree(input_data[index[midn]], kdtree(input_data[left], depth = depth+1), kdtree(input_data[right], depth = depth+1))
end

function kdfind(input_point::Vector{<:Real}, kdtree::Union{Kdtree, Nothing}; depth = 1)
    if isnothing(kdtree)
        return [NaN, NaN], Inf
    end
    dims = length(kdtree.location)
    sa = (depth+2) % dims + 1
    #subtree = deepcopy(kdtree)
    #search_path = Vector{Vector{<:Real}}()
    if (input_point[sa] <= kdtree.location[sa]) 
        nearestPoint, nearestDistance = kdfind(input_point, kdtree.left, depth = depth + 1)
    elseif (input_point[sa] > kdtree.location[sa]) 
        nearestPoint, nearestDistance = kdfind(input_point, kdtree.right, depth = depth + 1)
    end
    nowDistance = norm(input_point - kdtree.location)
    if nowDistance < nearestDistance
        nearestDistance = nowDistance
        nearestPoint = deepcopy(kdtree.location)
    end
    splitDistance = abs(input_point[sa] - kdtree.location[sa])

    if splitDistance > nearestDistance
        return nearestPoint,nearestDistance
    else
        nextTree = input_point[sa]<=kdtree.location[sa] ? kdtree.right : kdtree.left
        nearPoint, nearDistance = kdfind(input_point, nextTree, depth = depth+1)
        if nearDistance < nearestDistance
            nearestDistance = nearDistance
            nearestPoint = deepcopy(nearPoint)
        end
    end
    return nearestPoint, nearestDistance
end

struct FrequencyData{T}
    name::Vector{T}
    frequency::Vector{<:Real}
end

function get_frequency(data::Vector{<:Any})
    name_vector = Vector{typeof(data[1])}()
    times_vector = Vector{Int64}()
    for i in data
        isfind = findall(x->x==i, name_vector)
        if isfind == []
            push!(name_vector, i)
            append!(times_vector, 1)
        else
            times_vector[isfind[1]] += 1
        end
    end
    frequency_vector = times_vector ./ length(data)
    return FrequencyData(name_vector, frequency_vector)
end

function get_frequency(data, target)
    goal = findall(x->x==target, data)
    return length(goal)/length(data)
end

function naive_bayesian(dataset::DataFrame, input_data)
    namelist = names(dataset)
    n = length(input_data)
    goal_fre = get_frequency(dataset[!, namelist[end]])
    goal_gro = groupby(df, namelist[end])
    fre_matrix = hcat(zeros(Float64, length(goal_gro), n), goal_fre.frequency)
    for i in 1:length(goal_gro)
        df = goal_gro[i]
        for j in 1:n 
            fre_matrix[i, j] = get_frequency(df[!, namelist[j]], input_data[j])
        end
    end
    return [reduce(*, fre_matrix[i, :]) for i in 1:length(fre_matrix[:, 1])], goal_fre.name
end