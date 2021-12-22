

import pandas as pd
import numpy as np
import os
from typing import List
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


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

    df = df.copy()

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

    if mode == 'tsne':
        _plot_tsne(df=df, kernel=kernel, label_col=target_variable, to_screen=to_screen, to_file=to_file, export_folder=export_folder)


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

    amount_of_plots = len(kernel.get_kernel_feature_names()) * len(kernel.kernel_input_features)

    amount_of_plot_cols = int(np.round(np.sqrt(amount_of_plots)))
    amount_of_plot_rows = int(np.ceil(np.sqrt(amount_of_plots)))

    fig, ax = plt.subplots(amount_of_plot_rows, amount_of_plot_cols)

    if amount_of_plot_cols * amount_of_plot_rows == 1:
        ax_list = [ax]
    else:
        ax_list = ax.reshape(-1)

    ax_index = 0
    for input_feature in kernel.kernel_input_features:
        for kernel_feature in kernel.get_kernel_feature_names():
            _plot_scree_on_ax_2D(ax_list[ax_index], df_with_kernel, input_feature, kernel_feature, label_col)

            if kernel.x_decision_boundary is not None:
                _plot_boundary_on_ax_2D(ax_list[ax_index], input_feature, kernel_feature, kernel)

            ax_index = ax_index + 1

    fig.suptitle('Kernel {} with Performance {}'.format(kernel.get_kernel_name(), kernel.kernel_quality))

    #plt.show(block=True)
    _finalize_plot(kernel, fig, to_screen=to_screen, to_file=to_file, export_folder=export_folder)


def _plot_scree_3D(df, kernel:Abstract_Duovariate_Kernel, label_col:str
                   , to_screen: bool = True
                   , to_file: bool = False
                   , export_folder: str = False
                   ):
    df_with_kernel = kernel.apply(df)

    amount_of_plots = len(kernel.get_kernel_feature_names())

    amount_of_plot_cols = int(np.round(np.sqrt(amount_of_plots)))
    amount_of_plot_rows = int(np.ceil(np.sqrt(amount_of_plots)))

    #fig, ax = plt.subplots(amount_of_plot_rows, amount_of_plot_cols, projection='3D')
    fig = plt.figure()
    ax_list = []
    i = 1
    for col_index in range(amount_of_plot_cols):
        for row_index in range(amount_of_plot_rows):
            ax = fig.add_subplot(amount_of_plot_rows, amount_of_plot_cols, i, projection='3d')
            i = i+1
            ax_list.append(ax)

    #if amount_of_plot_cols * amount_of_plot_rows == 1:
    #    ax_list = [ax]
    #else:
    #    ax_list = ax.reshape(-1)

    ax_index = 0

    for kernel_feature_out in kernel.get_kernel_feature_names():
        _plot_scree_on_ax_3D(ax_list[ax_index], df_with_kernel, kernel.kernel_input_features[0], kernel.kernel_input_features[1], kernel_feature_out, label_col)
        if kernel.x_decision_boundary is not None:
            _plot_boundary_on_ax_3D(ax_list[ax_index], kernel.kernel_input_features[0], kernel.kernel_input_features[1], kernel_feature_out, kernel)

        ax_index = ax_index + 1

    fig.suptitle('Kernel {} with Performance {}'.format(kernel.get_kernel_name(), kernel.kernel_quality))

    # plt.show(block=True)
    _finalize_plot(kernel, fig, to_screen=to_screen, to_file=to_file, export_folder=export_folder)


def _plot_tsne(df:pd.DataFrame, kernel:Abstract_Kernel
                   , label_col:str
                   , to_screen: bool = True
                   , to_file: bool = False
                   , export_folder: str = False):

    if len(df) > 2000:
        df.sample(n=2000)
    df_with_kernel = kernel.apply(df)

    all_features = kernel.kernel_input_features.copy()
    for f in kernel.get_kernel_feature_names():
        all_features.append(f)

    X = df_with_kernel[all_features].to_numpy(dtype=np.float64)
    y = df_with_kernel[label_col].to_numpy(dtype=np.float64)

    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(1, 1)

    fig.suptitle('Kernel {} with Performance {} TSNE'.format(kernel.get_kernel_name(), kernel.kernel_quality))

    ax.scatter(X_tsne[:,0].reshape(-1), X_tsne[:,1].reshape(-1), c=y.reshape(-1))

    _finalize_plot(kernel, fig, to_screen=to_screen, to_file=to_file, export_folder=export_folder)


def _plot_scree_on_ax_2D(ax:plt.Axes, df, feature_x1, feature_x2, label_col):
    x1 = df[feature_x1].values.reshape(-1)
    x2 = df[feature_x2].values.reshape(-1)

    y = df[label_col].values.reshape(-1)

    ax.scatter(x1,x2, c=y)

    ax.set_title('{x1} vs {x2}'.format(x1=feature_x1, x2=feature_x2))


def _plot_scree_on_ax_3D(ax:plt.Axes, df, feature_x1, feature_x2, feature_x3, label_col):


    x1 = df[feature_x1].values.reshape(-1)
    x2 = df[feature_x2].values.reshape(-1)
    x3 = df[feature_x3].values.reshape(-1)
    y = df[label_col].values.reshape(-1)

    ax.scatter3D(x1,x2,x3, c=y)

    ax.set_title('{x1} vs {x2} vs {x3}'.format(x1=feature_x1, x2=feature_x2, x3=feature_x3))


def _plot_boundary_on_ax_2D(ax:plt.Axes, feature_x1, feature_x2, kernel:Abstract_Kernel):

    x_decision = kernel.get_decision_boundary([feature_x1,feature_x2])

    x1 = x_decision[:,0]
    x2 = x_decision[:,1]

    if type(x1) != np.ndarray:
        x1 = x1.get()
        x2 = x2.get()

    ax.scatter(x1, x2, marker='.', color='red')


def _plot_boundary_on_ax_3D(ax:plt.Axes, feature_x1, feature_x2, feature_x3, kernel:Abstract_Kernel):

    x_decision = kernel.get_decision_boundary([feature_x1,feature_x2, feature_x3])

    x1 = x_decision[:,0]
    x2 = x_decision[:,1]
    x3 = x_decision[:,1]

    if type(x1) != np.ndarray:
        x1 = x1.get()
        x2 = x2.get()
        x3 = x3.get()

    ax.scatter3D(x1, x2,x3, marker='.', color='red')


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



def plot_ranking(kernel_list:List[Abstract_Kernel]
                 , to_screen: bool = True
                 , to_file: bool = False
                 , export_folder: str = False
                 ):

    if to_file and export_folder is None:
        raise Exception('Need to specify a folder for the exports')

    if to_file:
        try:
            os.makedirs(export_folder , exist_ok=True)
        except Exception as e:
            print('ERROR whilst trying to create export folder {}'.format(export_folder))
            raise e

        if not os.path.isdir(export_folder):
            raise Exception('Export folder is not a folder: {}'.format(export_folder))


    kernel_names = [k.get_kernel_name() for k in kernel_list]
    kernel_qualities = [k.kernel_quality for k in kernel_list]

    y_pos = np.flip(np.arange(len(kernel_qualities)))

    fig, ax = plt.subplots()

    ax.barh(y_pos, kernel_qualities)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels=kernel_names, fontdict={'fontsize': 8})
    ax.set_xlabel('Performance')
    ax.set_title('Kernel Quality Ranking')
    plt.tight_layout()

    if to_file:
        filename = 'performance_ranking.png'
        full_path = '{export_folder}/{filename}'.format(export_folder=export_folder, filename=filename)

        fig.savefig(full_path)

    if to_screen:
        plt.show(block=True)


