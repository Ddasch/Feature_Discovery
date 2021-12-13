import cupy as cp
import datetime

from featurediscovery.fitter.cupy_fitter import Logistic_Regression

from featurediscovery.fitter.sklearn_fitters import Linear_Scikit, Logistic_Scikit



def test_Logistic_Regression_Performance_vs_ridge_tiny():
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

    logistic = Logistic_Regression(fit_metric='gini')
    linear = Linear_Scikit(fit_metric='gini')

    start_logistic = datetime.datetime.now()
    for i in range(100):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()

    start_ridge = datetime.datetime.now()
    for i in range(100):
        linear = Linear_Scikit(fit_metric='gini')
        fit_improvement = linear.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_ridge = datetime.datetime.now()



    duration_ridge = end_ridge - start_ridge
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn ridge: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))




def test_Logistic_Regression_Performance_vs_ridge_2D_large():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(100):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    x = x_big
    y = y_big

    logistic = Logistic_Regression(fit_metric='gini')
    linear = Linear_Scikit(fit_metric='gini')

    start_logistic = datetime.datetime.now()
    for i in range(1000):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()

    start_ridge = datetime.datetime.now()
    for i in range(1000):
        linear = Linear_Scikit(fit_metric='gini')
        fit_improvement = linear.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_ridge = datetime.datetime.now()



    duration_ridge = end_ridge - start_ridge
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn ridge: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))




def test_Logistic_Regression_Performance_vs_ridge_100D_large():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(100):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    #x = x_big
    #y = y_big

    x = cp.random.normal(0.0, 1, (x_big.shape[0], 1000))
    x[:,0] = x_big[:,0]
    x[:,1] = x_big[:,1]
    y = y_big


    logistic = Logistic_Regression(fit_metric='gini')
    linear = Linear_Scikit(fit_metric='gini')

    start_ridge = datetime.datetime.now()
    for i in range(1000):
        linear = Linear_Scikit(fit_metric='gini')
        fit_improvement = linear.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_ridge = datetime.datetime.now()

    start_logistic = datetime.datetime.now()
    for i in range(1000):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        #assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()





    duration_ridge = end_ridge - start_ridge
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn ridge: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))



def test_Logistic_Regression_Performance_vs_sklogistic_1000D_large():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(100):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    #x = x_big
    #y = y_big

    x = cp.random.normal(0.0, 1, (x_big.shape[0], 1000))
    x[:,0] = x_big[:,0]
    x[:,1] = x_big[:,1]
    y = y_big


    start_sklearn = datetime.datetime.now()
    for i in range(1000):
        log_scikit = Logistic_Scikit(fit_metric='gini')
        fit_improvement = log_scikit.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_sklearn = datetime.datetime.now()

    start_logistic = datetime.datetime.now()
    for i in range(1000):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        #assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()



    duration_ridge = end_sklearn - start_sklearn
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn logistic: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))



def test_Logistic_Regression_Performance_vs_sklogistic_100D_large():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(100):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    #x = x_big
    #y = y_big

    x = cp.random.normal(0.0, 1, (x_big.shape[0], 100))
    x[:,0] = x_big[:,0]
    x[:,1] = x_big[:,1]
    y = y_big


    start_sklearn = datetime.datetime.now()
    for i in range(1000):
        log_scikit = Logistic_Scikit(fit_metric='gini')
        fit_improvement = log_scikit.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_sklearn = datetime.datetime.now()

    start_logistic = datetime.datetime.now()
    for i in range(1000):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        #assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()



    duration_ridge = end_sklearn - start_sklearn
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn logistic: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))



def test_Logistic_Regression_Performance_vs_sklogistic_20D_large():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(100):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    #x = x_big
    #y = y_big

    x = cp.random.normal(0.0, 1, (x_big.shape[0], 20))
    x[:,0] = x_big[:,0]
    x[:,1] = x_big[:,1]
    y = y_big


    start_sklearn = datetime.datetime.now()
    for i in range(100):
        log_scikit = Logistic_Scikit(fit_metric='gini')
        fit_improvement = log_scikit.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_sklearn = datetime.datetime.now()

    import time
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()



    start_logistic = datetime.datetime.now()
    for i in range(100):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        #assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu


    duration_ridge = end_sklearn - start_sklearn
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn logistic: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))
    print('t_cpu: {}'.format(t_cpu))
    print('t_gpu: {}'.format(t_gpu))



def test_Logistic_Regression_Performance_vs_sklogistic_20D_huge():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(1000):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    #x = x_big
    #y = y_big

    x = cp.random.normal(0.0, 1, (x_big.shape[0], 20))
    x[:,0] = x_big[:,0]
    x[:,1] = x_big[:,1]
    y = y_big


    start_sklearn = datetime.datetime.now()
    for i in range(100):
        log_scikit = Logistic_Scikit(fit_metric='gini')
        fit_improvement = log_scikit.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_sklearn = datetime.datetime.now()

    import time
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()



    start_logistic = datetime.datetime.now()
    for i in range(100):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        #assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu


    duration_ridge = end_sklearn - start_sklearn
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn logistic: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))
    print('t_cpu: {}'.format(t_cpu))
    print('t_gpu: {}'.format(t_gpu))



def test_Logistic_Regression_Performance_vs_sklogistic_20D_extrahuge():
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

    x_big = x.copy()
    y_big = y.copy()

    for i in range(10000):
        x_big = cp.concatenate([x_big,x])
        y_big = cp.concatenate([y_big, y])

    #x = x_big
    #y = y_big

    x = cp.random.normal(0.0, 1, (x_big.shape[0], 20))
    x[:,0] = x_big[:,0]
    x[:,1] = x_big[:,1]
    y = y_big


    start_sklearn = datetime.datetime.now()
    for i in range(100):
        log_scikit = Logistic_Scikit(fit_metric='gini')
        fit_improvement = log_scikit.compute_fit_improvement(x, y)
        assert fit_improvement == 0.5
    end_sklearn = datetime.datetime.now()

    import time
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()



    start_logistic = datetime.datetime.now()
    for i in range(100):
        #print('iteration i={}'.format(i))
        logistic = Logistic_Regression(fit_metric='gini')
        fit_improvement = logistic.compute_fit_improvement(x, y)
        #assert fit_improvement == 0.5
    end_logistic = datetime.datetime.now()

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu


    duration_ridge = end_sklearn - start_sklearn
    duration_logistic = end_logistic - start_logistic

    print('Run time sklearn logistic: {}s {}ms'.format(duration_ridge.seconds, duration_ridge.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_logistic.seconds,
                                                           duration_logistic.microseconds))
    print('t_cpu: {}'.format(t_cpu))
    print('t_gpu: {}'.format(t_gpu))