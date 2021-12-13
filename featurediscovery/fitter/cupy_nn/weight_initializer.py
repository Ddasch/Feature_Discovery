import cupy as cp
import math



def init_2D_weights(shape:tuple, n_input:int, n_output:int, method:str):


    if method not in ['glorot' , 'xavier', 'glorot_norm' , 'xavier_norm', 'he']:
        raise Exception('Weight initializer {} not supported'.format(method))

    if method in ['glorot' , 'xavier']:
        return glorot_2D(shape,n_input)

    if method in ['glorot_norm' , 'xavier_norm']:
        return glorot_2D(shape,n_input)

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