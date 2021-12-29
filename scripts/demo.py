import numpy as np
import pandas as pd

from featurediscovery.kernel_search import evaluate_kernels


def generate_demo_dataset(no_samples:int):


    # Engineer a 2nd order polynomial solution
    x1 = np.random.uniform(-10, 10, no_samples)
    x2 = np.random.uniform(-10, 10, no_samples)

    poly2 = 0.1*x1*x1 - 10*x1*x2 + 2*x2*x2
    y_poly2 = np.zeros(no_samples)
    y_poly2[poly2 > 100] = 0.0

    #add some noise to pattern
    #x1 = x1 + np.random.randn(no_samples)
    #x2 = x2 + np.random.randn(no_samples)


    #engineer a cosine pattern
    x3 = np.random.uniform(-10, 10, no_samples)
    x4 = np.random.uniform(-10, 10, no_samples)

    cosine = np.cos(.5*x3 + .2*x4 + 10)
    y_cosine = np.zeros(no_samples)
    y_cosine[cosine > 0.6] = 0.6

    # add some noise to pattern
    #x3 = x3 + np.random.randn(no_samples)
    #x4 = x4 + np.random.randn(no_samples)


    # engineer some random noise features
    x_noise1 = np.random.uniform(-10, 10, (no_samples, 3))
    x_noise2 = np.random.uniform(-10, 10, (no_samples, 3))
    x_noise3 = np.random.uniform(-10, 10, (no_samples, 3))

    #create dataframe with the random data
    data = {}
    feature_index = 0


    data['x{}'.format(feature_index)] = x1
    feature_index = feature_index + 1
    data['x{}'.format(feature_index)] = x2
    feature_index = feature_index + 1

    data['x{}'.format(feature_index)] = x3
    feature_index = feature_index + 1
    data['x{}'.format(feature_index)] = x4
    feature_index = feature_index + 1


    for i in range(x_noise1.shape[1]):
        data['x{}'.format(feature_index)] = x_noise1[:,i]
        feature_index = feature_index + 1



    for i in range(x_noise2.shape[1]):
        data['x{}'.format(feature_index)] = x_noise2[:,i]
        feature_index = feature_index + 1



    for i in range(x_noise3.shape[1]):
        data['x{}'.format(feature_index)] = x_noise3[:,i]
        feature_index = feature_index + 1

    y = y_poly2 + y_cosine
    y[y > 0.5] = 1
    y[y < 0.5] = 0

    data['y'] = y

    df = pd.DataFrame(data=data)

    return df

if __name__ == '__main__':

    df = generate_demo_dataset(1000)

    #feature_space = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']
    feature_space = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
    evaluate_kernels(df
                     , target_variable='y'
                     , feature_space=feature_space
                     , monovariate_kernels=None
                     , duovariate_kernels=['poly2', 'poly3', 'rff_gauss']
                     , feature_standardizers=['minmax', 'raw', 'standard']
                     , plot_ranking=True
                     , eval_method='full'
                     , use_cupy='no'
                     )


    pass





