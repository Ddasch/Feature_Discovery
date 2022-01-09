# Copyright 2021-2022 by Frederik Christiaan Schadd
# All rights reserved
#
# Licensed under the GNU Lesser General Public License version 2.1 ;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/LGPL-2.1
#
# or consult the LICENSE file included in the project.


import os
import pandas as pd
from typing import List

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel

def export_kernel_ranking(kernel_list:List[Abstract_Kernel]
                          , export_folder:str
                          , export_format:str
                          , suffix:str = ''
                          ):

    if export_folder is None:
        raise Exception('No export folder provided')

    if export_format not in ['csv', 'json']:
        raise Exception('Unsupported export format {}'.format(export_format))

    try:
        os.makedirs(export_folder, exist_ok=True)
    except Exception as e:
        print('ERROR whilst trying to create export folder {}'.format(export_folder))
        raise e

    if not os.path.isdir(export_folder):
        raise Exception('Export folder is not a folder: {}'.format(export_folder))


    dict_list = []

    for kernel in kernel_list:
        d = {
            'Specific Kernel': kernel.get_kernel_name(),
            'Input': str(kernel.kernel_input_features),
            'Type' : kernel.get_kernel_type(),
            'Standardizer': kernel.standardizer.get_standardizer_name(),
            'Performance': kernel.kernel_quality
        }
        dict_list.append(d)

    df = pd.DataFrame(data=dict_list)

    if export_format == 'csv':
        df.to_csv('{folder}/performances{suffix}.csv'.format(folder=export_folder, suffix=suffix), index=False)

    if export_format == 'json':
        df.to_json('{folder}/performances{suffix}.json'.format(folder=export_folder, suffix=suffix), orient='records', indent=2)