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
    left::Union{Type{Nothing}, Kdtree}
    right::Union{Type{Nothing}, Kdtree}
end

function kdtree(input_data::Vector{<:Vector{<:Real}}; depth::Int64 = 1)
    function get_depth(depth::Int64, dims::Int64)
        if depth%dims == 0
            return dims
        else
            return depth%dims
        end
    end
    if length(input_data) == 0
        return Nothing
    end
    data = reduce(hcat, input_data)
    dims = length(input_data[1])
    len = length(input_data)
    midn = ceil(Int64, len/2)
    i = get_depth(depth, dims)
    index = sortperm(data[i, :])
    left = index[1:midn-1]
    right = index[midn+1:len]
    Kdtree(input_data[index[midn]], kdtree(input_data[left], depth = depth+1), kdtree(input_data[right], depth = depth+1))
end
