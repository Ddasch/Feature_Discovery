import cupy as cp




def test_polyfit_api():
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

    fit = cp.polyfit(x,y.reshape(-1),deg=1)

    #NOTE: polyfit is only monovariate, aka useless

    print('')

