function get_e(y::Vector{<:Real}, w::Vector{<:Real}, v::Real)
    n = length(y); left_v = floor(Int64, v) + 1
    left_index = [i for i in 1:left_v]
    right_index = [i for i in left_v+1:n]
    left_rindex = findall(x->x==1, y[begin:left_v])
    right_rindex = findall(x->x==-1, y[left_v+1:end])
    r = length(left_rindex) + length(right_rindex)
    bool = r>=n/2
    r_index = vcat(left_rindex, right_rindex.+left_v)
    e_index = vcat(setdiff(right_index, right_rindex.+left_v), setdiff(left_index, left_rindex))
    if bool
        return sum(abs.(y[e_index].*w[e_index])), vcat([1 for i in 1:left_v],[-1 for i in left_v+1:n])
    else
        return sum(abs.(y[r_index].*w[r_index])), vcat([-1 for i in 1:left_v],[1 for i in left_v+1:n])
    end
end


function get_e(y::Vector{<:Real}, w::Vector{<:Real}, v::Vector{<:Real})
    v_out = v[1]
    e_out = get_e(y, w, v[1])[1]
    for iv in v
        e = get_e(y, w, iv)[1]
        if e < e_out
            v_out = iv
            e_out = e
        end
    end
    return get_e(y, w, v_out), v_out
end
            

function get_α(e::Real)
    return 1/2*(log((1-e)/e))
end

function update_w(w::Vector{<:Real}, α::Real, y::Vector{<:Real}, g::Vector{<:Real})
    Zm = sum(w.*exp.(-α.*y .* g))
    return w = w./Zm .* exp.(-α.*y.*g)
end


x = [i for i in 0:9]
y = vcat(repeat([1, -1], inner=3), [1, 1, 1, -1])
v_split = [-0.5+i for i in 0:10]

w1 = [0.1 for i in 0:9]
(e1, g1), v1 = get_e(y, w1, v_split)
α1 = get_α(e1)
w2 =  update_w(w1, α1, y, g1)

(e2, g2), v2 = get_e(y, w2, v_split)
α2 = get_α(e2)
w3 = update_w(w2, α2, y, g2)

(e3, g3), v3 = get_e(y, w3, v_split)
α3 = get_α(e3)
w4 =update_w(w3, α3, y, g3)

result = sign.(α1.*g1 + α2.*g2 + α3.*g3)