# Flux Basics
## Taking Gradients
using Flux
f(x) = 3x^2 + 2x + 1
df(x) = gradient(f, x)[1]
@show df(2)
d2f(x) = gradient(df, x)[1]
@show d2f(2)

f(x, y) = sum((x .- y).^2)
@show gradient(f, [2, 1], [2, 0])

x = [2, 1]; y = [2, 0]
gs = gradient(params(x, y)) do
    f(x, y)
end
@show gs[x], gs[y]


## Building Simple Models
W = rand(2, 5)
b = rand(2)
predict(x) = W*x .+ b
function loss(x, y)
    ŷ = predict(x)
    sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2)
@show loss(x, y)

W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1*x .+ b1

W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2*x .+ b2

model(x) = layer2(σ.(layer1(x)))
@show model(rand(5))