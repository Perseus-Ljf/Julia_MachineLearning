using GLMakie
using Revise
using LinearAlgebra
includet("MachineLearning.jl")
include("MachineLearning.jl")

a = [[2, 3], [4, 7], [5, 4], [9, 6], [8, 1], [7, 2]]
kt = kdtree(a)
@show kdfind([5.5, 5], kt)

a = 1
b = a==2 ?  1 : 2