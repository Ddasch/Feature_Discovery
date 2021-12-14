
import pandas as pd







def _generate_all_list_combinations(**kargs):
    '''
    Helper method which generates all unique combinations of values in the input lists.
    :param kargs:
    :return: list of all unique combinations
    '''

    all_keys = list(kargs.keys())
    for k in all_keys:
        if type(kargs[k]) != list:
            raise Exception('input argument {} needs to be a list'.format(k))

    all_join_dfs = []

    for k in all_keys:
        param_df = pd.DataFrame(data={
            k: kargs[k],
            'join_key': 1
        })
        all_join_dfs.append(param_df)


    start_df = all_join_dfs[0]

    for i in range(1, len(all_join_dfs)):
        df_to_join = all_join_dfs[i]
        start_df = pd.merge(start_df,  df_to_join, on='join_key', how='inner')

    start_df = start_df.drop(columns=['join_key'])
    settings_list = [s for s in start_df.T.to_dict().values()]
    return settings_list