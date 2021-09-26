using GLMakie
using Revise
using LinearAlgebra
includet("MachineLearning.jl")
include("MachineLearning.jl")

a = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2],[3,4],[4,2],[5,3],[2,3],[7,5],[2,1]]
aa = reduce(hcat, a)
sortperm(aa[1,:])

kt = kdtree(a)

@show kdfind([1.3,2.4], kt)

