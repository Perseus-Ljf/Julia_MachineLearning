using GLMakie
using Revise
includet("MachineLearning.jl")
include("MachineLearning.jl")

test_data = [[2,2,4,-1],[2,3,7,-1],[1,1,3,1],[0,1,-2,1]]

w, b = perceptron(test_data, .01)
fig, ax, sca = scat(test_data)
pplot!(ax, w, b, 0:2, 0:2)
fig
