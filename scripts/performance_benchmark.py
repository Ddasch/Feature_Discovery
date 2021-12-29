import numpy as np
import cupy as cp
import pandas as pd
import datetime
import matplotlib.pyplot as plt


from featurediscovery import kernel_search


def get_test_dataset(n_samples:int, n_dimensions:int) -> pd.DataFrame:
    '''
    Get a test dataset with a quadratic solution baked in.
    :param n_samples:
    :param n_dimensions:
    :return:
    '''

    print('Start Generating dataset...')
    x = np.array([
        np.array([-3, 0, 1]),
        np.array([-2, 1, 2]),
        np.array([-1, 0, 3]),
        np.array([0, 1, 1]),
        np.array([1, 0, 2]),
        np.array([2, 1, 3]),
        np.array([3, 0, 1]),
        np.array([-4, 1, 2]),
        np.array([4, 0, 3]),
        np.array([-5, 1, 1]),
        np.array([5, 0, 2]),
        np.array([-6, 1, 3]),
        np.array([6, 0, 1])
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    x_big = x.copy()
    y_big = y.copy()

    #stretch dataset depending on desired n_samples
    n_concat = int(n_samples / x.shape[0]) - 1

    for i in range(n_concat):
        x_big = np.concatenate([x_big, x])
        y_big = np.concatenate([y_big, y])



    #add noise columns depending on amount of desired extra columns
    x = np.random.normal(0.0, 1, (x_big.shape[0], n_dimensions))
    x[:, 0] = x_big[:, 0]
    y = y_big

    data = {}
    for i in range(x.shape[1]):
        data['x{}'.format(i+1)] = x[:,i]
    data['y'] = y

    print('End Generating dataset.')
    return pd.DataFrame(data=data)




def benchmark(df:pd.DataFrame, n_test:int=100):
    start_sklearn = datetime.datetime.now()
    for i in range(n_test):
        print('Starting CPU test {}'.format(i + 1))
        results = kernel_search._search(df, feature_space=[x for x in df.columns if x!='y'], target_variable='y',
                                        duovariate_kernels=['poly2'],
                                        feature_standardizers=['raw'],
                                        eval_method='full', use_cupy='no')
    end_sklearn = datetime.datetime.now()

    import time
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()

    start_logistic = datetime.datetime.now()
    for i in range(n_test):
        print('Starting GPU test {}'.format(i+1))
        results = kernel_search._search(df, feature_space=[x for x in df.columns if x!='y'], target_variable='y',
                                        duovariate_kernels=['poly2'],
                                        feature_standardizers=['raw'],
                                        eval_method='full', use_cupy='yes')
    end_logistic = datetime.datetime.now()

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu

    duration_cpu = end_sklearn - start_sklearn
    duration_gpu = end_logistic - start_logistic

    print('Run time sklearn logistic: {}s {}ms'.format(duration_cpu.seconds, duration_cpu.microseconds))
    print('Run time cupy_nn logistic: {}s {}ms'.format(duration_gpu.seconds,
                                                       duration_gpu.microseconds))
    print('t_cpu: {}'.format(t_cpu))
    print('t_gpu: {}'.format(t_gpu))

    return (duration_cpu.seconds + duration_cpu.microseconds/1000000)/n_test \
            , (duration_gpu.seconds + duration_gpu.microseconds/1000000)/n_test


def benchmark_dim_scaling(n_test:int = 1):
    labels = []
    t_cpu = []
    t_gpu = []

    # test 0
    n_sample = 100
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('d={}'.format(n_dim))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    #test 1
    n_sample = 100
    n_dim = 5
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('d={}'.format(n_dim))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)


    # test 2
    n_sample = 100
    n_dim = 10
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('d={}'.format(n_dim))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    # test 3
    n_sample = 100
    n_dim = 20
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('d={}'.format(n_dim))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

  
    # test 4
    n_sample = 100
    n_dim = 30
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('d={}'.format(n_dim))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    '''
    # test 5
    n_sample = 100
    n_dim = 100
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('d={}'.format(n_dim))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)
    '''


    #make a bar plot of the performances
    bar_label_cpu = ['{:.2f}'.format(x) for x in t_cpu]
    bar_label_gpu = ['{:.2f}'.format(x) for x in t_gpu]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects_cpu = ax.bar(x - width / 2, t_cpu, width, label='CPU')
    rects_gpu = ax.bar(x + width / 2, t_gpu, width, label='GPU')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('avg runtime (s)')
    ax.set_title('Feature Discovery Runtime Benchmark , n={}'.format(n_sample))
    ax.set_xticks(x, labels)
    ax.legend()

    min_y = min(min(t_cpu), min(t_gpu))
    max_y = max(max(t_cpu), max(t_gpu))

    for rect, label in zip(rects_cpu, bar_label_cpu):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + max_y / 100, label, ha="center", va="bottom"
        )

    for rect, label in zip(rects_gpu, bar_label_gpu):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + max_y / 100, label, ha="center", va="bottom"
        )

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    fig.savefig('./runtime_dim_scaling.png')

    plt.show()



def benchmark_sample_scaling(n_test:int = 1):
    labels = []
    t_cpu = []
    t_gpu = []

    # test 0
    n_sample = 100
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    #test 1
    n_sample = 1000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    # test 2
    n_sample = 10000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    # test 3
    n_sample = 25000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    # test 4
    n_sample = 100000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    # test 5
    n_sample = 250000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    # test 6
    n_sample = 1000000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)

    '''
    # test 7
    n_sample = 500000
    n_dim = 2
    df = get_test_dataset(n_sample, n_dim)
    current_cpu_time, current_gpu_time = benchmark(df, n_test=n_test)
    labels.append('n={}'.format(n_sample))
    t_cpu.append(current_cpu_time)
    t_gpu.append(current_gpu_time)
    '''

    # make a bar plot of the performances
    bar_label_cpu = ['{:.2f}'.format(x) for x in t_cpu]
    bar_label_gpu = ['{:.2f}'.format(x) for x in t_gpu]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects_cpu = ax.bar(x - width / 2, t_cpu, width, label='CPU')
    rects_gpu = ax.bar(x + width / 2, t_gpu, width, label='GPU')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('avg runtime (s)')
    ax.set_title('Feature Discovery Runtime Benchmark - d={}'.format(n_dim))
    ax.set_xticks(x, labels)
    ax.legend()

    min_y = min(min(t_cpu), min(t_gpu))
    max_y = max(max(t_cpu), max(t_gpu))

    for rect, label in zip(rects_cpu, bar_label_cpu):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + max_y / 100, label, ha="center", va="bottom"
        )

    for rect, label in zip(rects_gpu, bar_label_gpu):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + max_y / 100, label, ha="center", va="bottom"
        )

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    fig.savefig('./runtime_sample_scaling.png')


if __name__ == '__main__':

    benchmark_dim_scaling(n_test=20)
    benchmark_sample_scaling(n_test=20)
