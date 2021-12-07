import cupy as cp
import pandas as pd



def test_cupy_array_create():
    df_iris = pd.read_csv('./datasets/iris.csv', sep=',')

    iris_np = df_iris[["sepal.length","sepal.width","petal.length","petal.width"]].values

    iris_cupy = cp.array(iris_np)

    print('')






