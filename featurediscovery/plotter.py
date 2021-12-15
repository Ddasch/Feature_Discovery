

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.kernels.monovariate.abstract_monovariate import Abstract_Monovariate_Kernel
from featurediscovery.kernels.duovariate.abstract_duovariate import Abstract_Duovariate_Kernel

def plot_kernel(df:pd.DataFrame, kernel:Abstract_Kernel, target_variable:str, mode:str = 'scree'):

    if mode not in ['scree', 'tsne']:
        raise Exception('Unsupported plot mode {}'.format(mode))

    if mode == 'scree':
        _plot_scree(df=df, kernel=kernel, label_col=target_variable)




def _plot_scree(df, kernel:Abstract_Kernel, label_col:str):
    if isinstance(kernel, Abstract_Monovariate_Kernel):
        _plot_scree_2D(df, kernel, label_col=label_col)

    if isinstance(kernel, Abstract_Duovariate_Kernel):
        _plot_scree_3D(df, kernel, label_col=label_col)

def _plot_scree_2D(df, kernel:Abstract_Monovariate_Kernel, label_col:str):

    df_with_kernel = kernel.apply(df)

    amount_of_plots = len(kernel.kernel_features) * len(kernel.features)

    amount_of_plot_cols = int(np.round(np.sqrt(amount_of_plots)))
    amount_of_plot_rows = int(np.ceil(np.sqrt(amount_of_plots)))

    fig, ax = plt.subplots(amount_of_plot_rows, amount_of_plot_cols)

    if amount_of_plot_cols * amount_of_plot_rows == 1:
        ax_list = [ax]
    else:
        ax_list = ax.reshape(-1)

    ax_index = 0
    for input_feature in kernel.features:
        for kernel_feature in kernel.kernel_features:
            _plot_scree_on_ax_2D(ax_list[ax_index], df_with_kernel, input_feature, kernel_feature, label_col)
            ax_index = ax_index + 1

    fig.suptitle('Kernel {} with Performance {}'.format(kernel.get_kernel_name(), kernel.kernel_quality))

    plt.show(block=True)


def _plot_scree_3D(df, kernel:Abstract_Kernel, label_col:str):
    pass




def _plot_scree_on_ax_2D(ax, df, feature_x1, feature_x2, label_col):


    x1 = df[feature_x1].values.reshape(-1)
    x2 = df[feature_x2].values.reshape(-1)

    y = df[label_col].values.reshape(-1)

    ax.scatter(x1,x2, c=y)

    ax.set_title('{x1} vs {x2}'.format(x1=feature_x1, x2=feature_x2))





