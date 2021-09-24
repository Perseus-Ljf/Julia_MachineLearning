using GLMakie
using Revise
includet("MachineLearning.jl")
include("MachineLearning.jl")

a = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
aa = reduce(hcat, a)
sortperm(aa[1,:])

kdtree(a)

