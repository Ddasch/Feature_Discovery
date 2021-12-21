

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.kernels.monovariate.abstract_monovariate import Abstract_Monovariate_Kernel
from featurediscovery.kernels.duovariate.abstract_duovariate import Abstract_Duovariate_Kernel

def plot_kernel(df:pd.DataFrame
                , kernel:Abstract_Kernel
                , target_variable:str
                , mode:str = 'scree'
                , to_screen:bool=True
                , to_file:bool=False
                , export_folder:str=False
                ):

    if not to_file and not to_screen:
        raise Warning('WARNING invoking plot routine whilst having to_screee and to_file both set as False')

    if mode not in ['scree', 'tsne']:
        raise Exception('Unsupported plot mode {}'.format(mode))

    if to_file  and export_folder is None:
        raise Exception('Need to specify a folder for the exports')

    if to_file:
        try:
            os.makedirs(export_folder + '/figures/', exist_ok=True)
        except Exception as e:
            print('ERROR whilst trying to create export folder {}'.format(export_folder))
            raise e

        if not os.path.isdir(export_folder):
            raise Exception('Export folder is not a folder: {}'.format(export_folder))

    if mode == 'scree':
        _plot_scree(df=df, kernel=kernel, label_col=target_variable, to_screen=to_screen, to_file=to_file, export_folder=export_folder)




def _plot_scree(df, kernel:Abstract_Kernel, label_col:str
                , to_screen: bool = True
                , to_file: bool = False
                , export_folder: str = False
                ):
    if isinstance(kernel, Abstract_Monovariate_Kernel):
        _plot_scree_2D(df, kernel, label_col=label_col, to_screen=to_screen, to_file=to_file, export_folder=export_folder)

    if isinstance(kernel, Abstract_Duovariate_Kernel):
        _plot_scree_3D(df, kernel, label_col=label_col, to_screen=to_screen, to_file=to_file, export_folder=export_folder)

def _plot_scree_2D(df, kernel:Abstract_Monovariate_Kernel, label_col:str
                   , to_screen: bool = True
                   , to_file: bool = False
                   , export_folder: str = False
                   ):

    df_with_kernel = kernel.apply(df)

    amount_of_plots = len(kernel.get_kernel_feature_names()) * len(kernel.features)

    amount_of_plot_cols = int(np.round(np.sqrt(amount_of_plots)))
    amount_of_plot_rows = int(np.ceil(np.sqrt(amount_of_plots)))

    fig, ax = plt.subplots(amount_of_plot_rows, amount_of_plot_cols)

    if amount_of_plot_cols * amount_of_plot_rows == 1:
        ax_list = [ax]
    else:
        ax_list = ax.reshape(-1)

    ax_index = 0
    for input_feature in kernel.features:
        for kernel_feature in kernel.get_kernel_feature_names():
            _plot_scree_on_ax_2D(ax_list[ax_index], df_with_kernel, input_feature, kernel_feature, label_col)
            ax_index = ax_index + 1

    fig.suptitle('Kernel {} with Performance {}'.format(kernel.get_kernel_name(), kernel.kernel_quality))

    #plt.show(block=True)
    _finalize_plot(kernel, fig, to_screen=to_screen, to_file=to_file, export_folder=export_folder)


def _plot_scree_3D(df, kernel:Abstract_Kernel, label_col:str
                   , to_screen: bool = True
                   , to_file: bool = False
                   , export_folder: str = False
                   ):
    pass




def _plot_scree_on_ax_2D(ax:plt.Axes, df, feature_x1, feature_x2, label_col):


    x1 = df[feature_x1].values.reshape(-1)
    x2 = df[feature_x2].values.reshape(-1)

    y = df[label_col].values.reshape(-1)

    ax.scatter(x1,x2, c=y)

    ax.set_title('{x1} vs {x2}'.format(x1=feature_x1, x2=feature_x2))




def _finalize_plot(kernel:Abstract_Kernel
                   , fig:plt.Figure
                   , to_screen: bool = True
                   , to_file: bool = False
                   , export_folder: str = False
                   ):

    if to_file:
        filename = '{qual} {kernelname}.png'.format(qual=kernel.kernel_quality, kernelname=kernel.get_kernel_name())
        full_path = '{export_folder}/figures/{filename}'.format(export_folder=export_folder, filename=filename)

        fig.savefig(full_path)

    if to_screen:
        plt.show(block=True)



