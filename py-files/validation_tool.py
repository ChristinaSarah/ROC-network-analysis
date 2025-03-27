'''
Set of functions to import dataset and perform data quality check before
running analysis, including filtering and manipulating the raw final
dataset.
'''

import pandas as pd
import os


def get_dataframe_into_ds(path: str,
                          file_name: str,
                          is_xlsx: bool) -> pd.DataFrame:

    '''
    Import dataframe
    '''

    os.chdir(path)

    if is_xlsx:
        ds = pd.read_excel(file_name)
    else:
        ds = pd.read_csv(file_name)
    final_ds = pd.DataFrame(ds)

    os.chdir("/workspaces/ROC-network-analysis")
    ##os.chdir("/Users/andrealunghini/network-analysis/")

    return final_ds


def check_data_quality(ds: pd.DataFrame):

    '''
    Performs initial datachecks
    '''

    for u in [e for e in ds.columns.to_list() if "friend_" not in e]:
        print("variable: ", u , ", count_unique_values:" , ds[u].nunique())


def assure_data_quality(ds: pd.DataFrame,
                        target_variables:list):

    '''
    Problem: mix of data types: IDs should be string or float
    globally.
    Solution: for all the ID variables to be float.
    '''

    for tg in target_variables:
        ds[tg] = ds[tg].apply(lambda x: float(x))
        ds = pd.DataFrame(ds)

    return ds

#def get_peers_outside_class_warning(ds: pd.DataFrame):

#    '''
#    Every student can state a friendship only with a peer
#    in the same class. This function spits out the class_id and
#    the student_id of the student stating to be friend with another
#    peer but from a different class.
#    Main purpose: data quality check.
#    '''

#    import math

#    for c in ds.classroom_id.unique():
#        class_ = ds[ds['classroom_id']==c]

#        sutdentIdsKeys= [s for s in class_.student_id.unique()]
#        studentDct = dict(zip(sutdentIdsKeys, [None]*len(sutdentIdsKeys)))

#       class_composition = [s for s in class_.student_id.unique()]
#        num_nodes=[1,2,3]

#        for s in class_composition:
#            FlagCount = 0
#            for n in num_nodes:
#                ff = class_[class_['student_id']==s][f'friend_{n}'].to_list()[0]
#                # If a peer does not state all 3 friends, the nan value counted as
#                # outside class if not negated in if condition.
#                if ff not in class_composition and not math.isnan(ff):
#                    FlagCount +=1
#            if FlagCount>0:
#                studentDct[s] = True
#            else:
#                studentDct[s] = False
#
#        value_to_find = True
#        keys = [key for key, value in studentDct.items() if value == value_to_find]
#
#        if len(keys)>0:
#            print("Assigned class: ", c, "Students with friends out class: ", keys)
#            warning_flag=True
#        else:
#            warning_flag =False
#
#    return warning_flag
