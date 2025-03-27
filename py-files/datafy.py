'''
Set of functions to run statistical aggregated analysis and the
final manipulated dataset, after validation tools have been run.
'''

import pandas as pd


def get_isolated_child_inwards_per_class(ds: pd.DataFrame,
                                      #get_peers_outside_class_warning,
                                      target_variable:str):

    '''
    Computes the fraction of isolated peers per class level.
    Also added functionality to specify target_variable: this
    is an aggregation variable. Can act on `friend_` or on 
    `support_` for example.
    Please check README file to get more details.
    '''

    #w = get_peers_outside_class_warning(ds)

    #if w == True:
    #   print("Please be aware that peers have nominated others outside class.")
    #   print("The calculations will be performed anyways. Check input dataset.")

    import numpy as np
    import itertools
    import pandas as pd

    num_nodes = [1,2,3]
    isolated_students_list = []

    classIdsKeys= [c for c in ds.classroom_id.unique()]
    classDct = dict(zip(classIdsKeys, [None]*len(classIdsKeys)))

    for c in ds.classroom_id.unique():
        isolated_peer_count = 0
        class_ = ds[ds['classroom_id']==c]
        # Assumed that each student_id is unique per row
        class_composition = [s for s in class_.student_id.unique()]
        class_size = len(class_composition)

        for s in class_composition:
                fl = []
                for n in num_nodes:
                        i = class_[class_['student_id']!=s][f'{target_variable}{n}'].dropna().to_list()
                        fl.append(i)

                other_peers = set(list(itertools.chain(*fl)))

                ## Making sure isolation is defined per class.
                if s not in other_peers and float(str(c)[3])==float(str(s)[3]):
                        isolated_students_list.append(s)
                        isolated_peer_count +=1
            # Already percentages.
        classDct[c] = np.round(isolated_peer_count/class_size,6)

    isolated_frak_ds = pd.DataFrame({'classroom_id': classDct.keys(),'isolated_f_%': classDct.values()})

    isolated_frak_ds = pd.DataFrame({'classroom_id': classDct.keys(),'isolated_f_%': classDct.values()})

    isolated_student_id_df = pd.DataFrame(isolated_students_list,columns=['student_id'])
    isolated_student_id_df[f'inwards_isolation_flag_{target_variable}']=1

    return isolated_frak_ds,isolated_student_id_df


def get_paired_ds(ds:pd.DataFrame,
                  target_variable:str) -> pd.DataFrame:

    '''
    Pick each student_id (start_node) and couple it with each friend_*
    excluding null friend_*.
    '''

    import numpy as np
    import math
    import itertools
    import pandas as pd

    num_nodes = [1,2,3]
    #Indeed a pair but need to track class.
    start_end_nodes_ds = []

    for _, row in ds.iterrows():
        start_node = row['student_id']
        c = row['classroom_id']

        for n in num_nodes:
            end_node = [row[f'{target_variable}{n}']][0]

            #To be pair IFF end node is not void.
            if not math.isnan(end_node):
                triple = (c,start_node,end_node)
                start_end_nodes_ds.append(triple)

    start_end_nodes_ds = pd.DataFrame(start_end_nodes_ds,columns=['classroom_id','start_node','end_node'])

    return start_end_nodes_ds


def get_reciprocal_friendship_ds(paired_ds: pd.DataFrame) -> pd.DataFrame:

    '''
    Take as input the parsed friendship pairs and spits out
    a dataset with reciprocal friendships (one per row which
    means cardinality is doubled per each row).
    '''

    import pandas as pd

    # Find reciprocal pairs
    reciprocal_pairs = []
    visited = set()

    for _, row in paired_ds.iterrows():
        start_node = row['start_node']
        end_node = row['end_node']
        pair = (start_node, end_node)
        reverse_pair = (end_node, start_node)

        if reverse_pair in visited:
            reciprocal_pairs.append(pair)

        visited.add(pair)

    reciprocal_pairs_df = pd.DataFrame(reciprocal_pairs, columns=['start_node', 'end_node'])
    reciprocal_pairs_df = reciprocal_pairs_df.merge(paired_ds, on=['start_node', 'end_node'])
    reciprocal_pairs_df = reciprocal_pairs_df[['classroom_id','start_node','end_node']]

    # Assuming that each individual does not write more than one a friend at each row, given
    # reciprocity.
    reciprocal_pairs_df['cardinality'] = 2

    return reciprocal_pairs_df


def get_reciprocity_total_nominations_frak(paired_ds: pd.DataFrame,
                                           reciprocity_ds: pd.DataFrame
                                           ) -> pd.DataFrame:

    '''
    Take as input the paired and reciprocity datasets and såits out
    the total number of nominations as well as the total number of reciprocal
    nomination, using method 1. Please read README file.
    Aggegation per class
    '''

    import pandas as pd

    total_nominations = paired_ds.groupby(['classroom_id'])['start_node'].count()
    total_reciprocity = reciprocity_ds.groupby(['classroom_id'])['cardinality'].sum()

    class_size_df = pd.merge(total_nominations,total_reciprocity,on=['classroom_id'],how='left')

    class_size_df['reciprocity_f']=class_size_df['cardinality']/class_size_df['start_node']

    new_column_names = {
    'start_node': 'tot_nominations_count',
    'cardinality': 'tot_reciprocity_count',
    'reciprocity_f': 'reciprocity_f'
    }

    class_size_df = class_size_df.rename(columns=new_column_names).reset_index(drop=False)

    class_size_df = class_size_df.fillna(0)

    return class_size_df


def get_isolated_outwards_info(ds: pd.DataFrame,
                         target_variable:str) -> pd.DataFrame:   

    import pandas as pd

    meas_dict = {
        "classroom_id": [],
        "student_count": [],
        "isolated_o": [],
        "isolated_share_o":[]
        }
    
    isolated_students_dict = {
    "classroom_id": [],
    "student_id": []
    }
    
    numb_nodes = [1, 2, 3]

    for class_id in list(ds.classroom_id.unique()):
        view_0, student_list= ds[ds.loc[:,'classroom_id']==class_id], list(ds[ds.loc[:,'classroom_id']==class_id].student_id.unique())
        isolated_count,student_count = 0,len(student_list)
        for s in student_list:
            view_1 = view_0[view_0.loc[:,'student_id']==s]
            if sum([[view_1[f'{target_variable}{n}'].isna() for n in numb_nodes][0].reset_index(drop=True)[0] for n in numb_nodes]) == max(numb_nodes):
                isolated_count += 1
                for keys in isolated_students_dict:
                    if keys =='classroom_id': 
                        isolated_students_dict[keys].append(class_id)
                    elif keys =='student_id':
                        isolated_students_dict[keys].append(s)

        for keys in meas_dict:
            if keys =='classroom_id': 
                meas_dict[keys].append(class_id)
            elif keys =='student_count': 
                meas_dict[keys].append(student_count)
            elif keys =='isolated_o': 
                meas_dict[keys].append(isolated_count)
            else:
                meas_dict[keys].append(0)

    output_df = pd.DataFrame.from_dict(meas_dict)
    output_df["isolated_share_o"]=output_df["isolated_o"]/output_df["student_count"]

    output_df_isolated_o = pd.DataFrame.from_dict(isolated_students_dict)

    return output_df,output_df_isolated_o
## END