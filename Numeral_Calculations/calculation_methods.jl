function gauss(A::Matrix{<:Real}, b::Vector{<:Real}; eps::Float64 = 1e-7)
    n = length(b)
    matrix = hcat(A, b)
    for i in 1:n-1
        index = findmax(abs.(matrix[i:end, i]))[2]
        if index != 1
            temp = matrix[i, i:end]
            matrix[i, i:end] = matrix[i+index-1, i:end] 
            matrix[i+index-1, i:end] = temp
        end
        for j in i+1:n
            matrix[j, i:end] = matrix[j, i:end]*matrix[i, i]/matrix[j, i] - matrix[i, i:end]
        end
    end
    x = zeros(Float64, n)
    x[n] = matrix[n, n+1]/matrix[n, n]
    for i in n-1:-1:1
        x[i] = (matrix[i, end]-x[i+1:end]'*matrix[i,i+1:n])/matrix[i, i]
    end 
    return x
end