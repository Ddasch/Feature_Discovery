import cupy as cp


from featurediscovery.fitter.sklearn_fitters import Linear_Scikit


def test_linear_ridge():
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

    linear = Linear_Scikit(fit_metric='gini')

    fit_improvement = linear.compute_fit_improvement(x,y)

    assert fit_improvement == 0.5
    print('')


