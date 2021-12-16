import cupy as cp
import math



def init_2D_weights(shape:tuple, n_input:int, n_output:int, method:str):


    if method not in ['glorot' , 'xavier', 'glorot_norm' , 'xavier_norm', 'he']:
        raise Exception('Weight initializer {} not supported'.format(method))

    if method in ['glorot' , 'xavier']:
        return glorot_2D(shape,n_input)

    if method in ['glorot_norm' , 'xavier_norm']:
        return glorot_norm_2D(shape,n_input, n_output)

    if method in ['he']:
        return he_2d(shape, n_input)


def glorot_2D(shape:tuple, n_input:int):

    range_min = -math.sqrt(1/n_input)
    range_max = math.sqrt(1/n_input)

    return cp.random.uniform(range_min,range_max, shape)


def glorot_norm_2D(shape:tuple, n_input:int, n_output:int):
    range_min = -math.sqrt(6 / (n_input + n_output))
    range_max = math.sqrt(6 / (n_input + n_output))

    return cp.random.uniform(range_min, range_max, shape)


def he_2d(shape:tuple, n_input:int):

    return cp.random.normal(0.0, math.sqrt(2/n_input), shape)


def covariance(X:cp.ndarray, Y:cp.ndarray):
    '''

    :param X: shape [n_sample, n_dim]
    :param Y: shape [n_sample, n_outputs]
    :return: cross covariances between X and Y in shape [n_dim, n_outputs]
    '''

    if len(X.shape) != 2:
        raise Exception('X must be of shape [n_sample, n_dim]')

    if len(Y.shape) != 2:
        raise Exception('Y must be of shape [n_sample, n_outs]')

    first_means = cp.mean(X, axis=0, keepdims=True)
    first_part = X - first_means

    second_means = cp.mean(Y, axis=0, keepdims=True)
    second_part = Y - second_means

    #mult = cp.multiply(first_part, second_part)
    mult = cp.einsum('ij,ik->ijk',first_part,second_part)

    cov = cp.mean(mult, axis=0, keepdims=True)

    cov = cov.reshape(X.shape[1], Y.shape[1])

    return cov