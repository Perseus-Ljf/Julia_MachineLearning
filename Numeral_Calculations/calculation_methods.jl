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

function doolittle(a::Matrix{<:Real})
    n = length(a[1, :])
    u = zeros(Float64, n, n)
    l = zeros(Float64, n, n)
    u[1, :] = a[1, :]
    l[:, 1] = a[:, 1] ./ u[1, 1]
    for r in 2:n
        for i in r:n
            u[r, i] = a[r, i] - l[r, 1:r-1]'*u[1:r-1, i]
            l[i, r] = (a[i, r] - l[i, 1:r-1]'*u[1:r-1, r])/u[r, r]
        end
    end
    return u, l
end

function crout(a::Matrix{<:Real})
    n = length(a[1, :])
    u = zeros(Float64, n, n)
    l = zeros(Float64, n, n)
    l[:, 1] = a[:, 1]
    u[1, :] = a[1, :]/l[1, 1]
    for r in 2:n
        for i in r:n
            l[i, r] = a[i, r] - l[i, 1:r-1]'*u[1:r-1, r]
            u[r, i] = (a[r, i] - l[r, 1:r-1]'*u[1:r-1, i])/l[r,r]
        end
    end
    return u, l
end

function downtri(A::Matrix{<:Real}, b::Vector{<:Real})
    n = length(b)
    x = zeros(Float64, n)
    x[1] = b[1] / A[1, 1]
    for i in 2:n
        x[i] = (b[i] - x[1:i-1]'*A[i, 1:i-1]) / A[i, i]
    end
    return x 
end

function uptri(A::Matrix{<:Real}, b::Vector{<:Real})
    n = length(b)
    x = zeros(Float64, n)
    x[n] = b[n] / A[n, n]
    for i in n-1:-1:1
        x[i] = (b[i] - x[i+1:n]'*A[i, i+1:n]) / A[i, i]
    end
    return x 
end

function lu_solve(A::Matrix{<:Real}, b::Vector{<:Real}; method = doolittle)
    u, l = method(A)
    @show u,l
    y = downtri(l, b)
    x = uptri(u, y)
    return x
end

function cholesky(a::Matrix{<:Real})
    n = length(A[1, :])
    l = zeros(Float64, n, n)
    l[1, 1] = sqrt(a[1, 1])
    l[2:end, 1] = a[2:end, 1]/l[1, 1]
    for j in 2:n-1
        for i in j+1:n
            l[j, j] = (a[j, j] - l[j, 1:j-1]'*l[j, 1:j-1])^(1/2)
            l[i, j] = (a[i, j] - l[i, 1:j-1]'*l[j, 1:j-1])/l[j, j]
        end
    end
    l[n, n] = (a[n, n] - l[n, 1:n-1]'*l[n, 1:n-1])^(1/2)
    return Matrix(l'), Matrix(l)
end