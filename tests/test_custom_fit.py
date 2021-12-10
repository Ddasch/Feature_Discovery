
import cupy as cp
from tests.nn_test_case2 import *


from featurediscovery.fitter.cupy_fitter import Linear_Cupy_SGD
from fitter.cupy_nn.models import SimpleModel
from fitter.cupy_nn.costs import CrossEntropyCost
from fitter.cupy_nn.layers import Layer


def test_layer():
    x = cp.array([
        cp.array([1, 2]),
        cp.array([1, 3]),
        cp.array([1, 4]),
        cp.array([2, 3]),
        cp.array([2, 4]),
        cp.array([3, 4]),
        cp.array([2, 1]),
        cp.array([3, 1]),
        cp.array([3, 2]),
        cp.array([4, 1]),
        cp.array([4, 2]),
        cp.array([4, 3])
    ])

    y = cp.array([0,0,0,0,0,0,1,1,1,1,1,1]).reshape([-1,1])

    layer = Layer(input_size=x.shape[1], layer_size=1, activation_func='sigmoid')

    X = x.transpose()
    res = layer.linear_activation_forward(X)


    print('')


def test_cost():

    cost = CrossEntropyCost()


    y = cp.array([1,1,1,1,0,0,0,0]).reshape((1,-1))

    y_hat_good = cp.array([1,1,1,1,0,0,0,0]).reshape((1,-1))
    y_hat_bad = cp.array([0,0,0,0,1,1,1,1]).reshape((1,-1))

    cost_good = cost.compute_cost(y_hat_good,y)
    cost_bad = cost.compute_cost(y_hat_bad,y)

    assert 0.001 > cost_good
    assert 20 < cost_bad

    Y, AL,_ = compute_cost_test_case()

    print("cost = " + str(cost.compute_cost(AL, Y)))


def test_linear_backwards():
    dZ, linear_cache = linear_backward_test_case()

    layer = Layer(input_size=3, layer_size=1, activation_func='sigmoid')

    layer.cache = linear_cache
    layer.W = linear_cache[1]
    layer.b = linear_cache[2]

    dA_prev, dW, db = layer.linear_backward(dZ)
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db))

    print('')

def test_linear_backwards2():
    AL, linear_activation_cache = linear_activation_backward_test_case()

    layer = Layer(input_size=3, layer_size=1, activation_func='sigmoid')

    layer.cache = linear_activation_cache[0]
    layer.W = linear_activation_cache[0][1]
    layer.b = linear_activation_cache[0][2]
    layer.full_cache = linear_activation_cache

    dA_prev, dW, db = layer.linear_activation_backward(AL)
    print("sigmoid:")
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db) + "\n")


def test_simple_fit():
    x = cp.array([
        cp.array([1, 2]),
        cp.array([1, 3]),
        cp.array([1, 4]),
        cp.array([2, 3]),
        cp.array([2, 4]),
        cp.array([3, 4]),
        cp.array([2, 1]),
        cp.array([3, 1]),
        cp.array([3, 2]),
        cp.array([4, 1]),
        cp.array([4, 2]),
        cp.array([4, 3])
    ])

    y = cp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape([-1, 1])

    model = SimpleModel()

    model.fit(x,y)

    y_hat, y_hat_prob = model.score(x)

    np.testing.assert_array_equal(y, y_hat)

    print('')