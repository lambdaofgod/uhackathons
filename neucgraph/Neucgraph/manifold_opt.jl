using Base: AbstractCartesianIndex
using Pkg
Pkg.activate(".")
Pkg.add("Zygote")
Pkg.build()
using Zygote
using Manifolds
using LinearAlgebra
using ManifoldDiff

M = Hyperbolic(2)
X = [[1.0 0.0 0.0]; [2.0 1.0 0.0]]
Y = [[2.0 1.0 0.0]; [1.0 0.0 0.0]]
ME = Euclidean(2)

R = rand(M, 2)

f(X, Y) = distance(M, X, Y)

g(X, Y) = distance(ME, X, Y)

# function hyperbolic_metric(X, Y)
#     -X[1] * Y[1] + dot(X[2:end], Y[2:end])
# end

hyperbolic_metric = [
    [-1.0 0.0 0.0];
    [0.0 1.0 0.0];
    [0.0 0.0 1.0]
]

x = X[1, :]
y = X[2, :]

hyperbolic_dist(x, y) = acosh(-(x' * hyperbolic_metric * y))

# algorithm 1 from https://arxiv.org/pdf/1806.03417.pdf

hyperbolic_dist(X[1, :], Y[1, :])

# this is gradient of distance w.r.t second parameter
manifolds_gradient_result = ManifoldDiff.grad_distance(M, X[1, :], Y[1, :], 1)


struct ManifoldWithMetricMatrix
    manifold::AbstractManifold
    metric::Matrix
end

HyperbolicWithMetric = ManifoldWithMetricMatrix(M, hyperbolic_metric)

function riemannian_gradient2(M::ManifoldWithMetricMatrix, f, x, y)
    (euc_grad_x, euc_grad_y) = Zygote.gradient(f, x, y)
    h = hyperbolic_metric * [euc_grad_x euc_grad_y]'
    #project(M.manifold, y, hyperbolic_metric .* euc_grads)
    h
end

zygote_gradient_result = riemannian_gradient2(HyperbolicWithMetric, hyperbolic_dist, X[1, :], Y[1, :])

@assert isapprox(zygote_gradient_result, manifolds_gradient_result)

riemannian_gradient2.(Ref(HyperbolicWithMetric), Ref(hyperbolic_dist), eachrow(X), eachrow(Y))

Zygote.gradient(hyperbolic_dist, X[1, :], Y[1, :])
