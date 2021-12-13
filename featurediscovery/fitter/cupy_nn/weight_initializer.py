import cupy as cp
import math



def init_2D_weights(shape:tuple, n_input:int, method:str):


    if method not in ['glorot' , 'xavier']:
        raise Exception('Weight initializer {} not supported'.format(method))

    if method in ['glorot' , 'xavier']:
        return glorot_2D(shape,n_input)


def glorot_2D(shape:tuple, n_input:int):

    range_min = -math.sqrt(1/n_input)
    range_max = math.sqrt(1/n_input)

    return cp.random.uniform(range_min,range_max, shape)

