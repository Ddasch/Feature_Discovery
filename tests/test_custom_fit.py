
import cupy as cp

from featurediscovery.fitter.cupy_fitter import Linear_Cupy_SGD, Layer


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