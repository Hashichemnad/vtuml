import pandas as pd
from pandas import DataFrame
df_tennis = pd.read_csv('id3.csv')
print("\nGiven Play Tennis Data Set:\n\n",df_tennis)
#df_tennis.columns[0]
df_tennis.keys()[4]
def entropy(probs):
    import math
    return sum([-prob*math.log(prob,2)for prob in probs])
def entropy_of_list(a_list):
    from collections import Counter
    cnt=Counter(x for x in a_list)
    num_instances=len(a_list)
    print("Number of Instances of the Current Sub Class is {0} :".format(num_instances))
    probs=[x/num_instances for x in cnt.values()]
    print("\n Classes:",min(cnt),max(cnt))
    print("\n Probability of Class {0} is {1}:".format(min(cnt),min(probs)))
    print("\n Probability of Class {0} is {1}:".format(max(cnt), max(probs)))
    return entropy(probs)
print("\n INPUT DATA SET FOR ENTROPY CALCULATION :\n",df_tennis['playtennis'])
total_entropy=entropy_of_list(df_tennis['playtennis'])
print("\n Total Entropy of playtennis Data Set:",total_entropy)
def information_gain(df,split_attribute_name,target_attribute_name,trace=0):
    print("Information Gain Calculation of",split_attribute_name)
    '''
    Takes a DataFrame of attributes,and quantifies the entropy of a target
    attribute after performing a split along the values of anotheri attribute.
    '''
    df_split = df.groupby(split_attribute_name)
    nobs=len(df.index)*1.0
    print("NOBS",nobs)
    df_agg_ent=df_split.agg({target_attribute_name :[entropy_of_list,lambda x:len(x)/nobs]})[target_attribute_name]
    print("DFAGGENT",df_agg_ent)
    df_agg_ent.columns=["Entropy",'PropObservations']
    new_entropy=sum(df_agg_ent['Entropy']*df_agg_ent['PropObservations'])
    old_entropy=entropy_of_list(df[target_attribute_name])
    return old_entropy-new_entropy
print("Info-gain for Outlook is:'+str(information_gain(df_tennis,'Outlook',playtennis')),\n")
print("Info-gain for Humidity is:'+str(information_gain(df_tennis,'Humidity',playtennis')),\n")
print("Info-gain for Wind is:'+str(information_gain(df_tennis,'Wind',playtennis')),\n")
print("Info-gain for Temperature is:'+str(information_gain(df_tennis,'Temperature',playtennis')),\n")
def id3(df,target_attribute_name,attribute_name,default_class=None):
    from collections import Counter
    cnt=Counter(x for x in df[target_attribute_name])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or(not attribute_names):
        return default_class
    else:
        default_class=max(cnt.keys())
        gainz=[information_gain(df,attr,target_attribute_name)for attr in attribute_names]
        index_of_max=gainz.index(max(gainz))
        best_attr=attribute_names[index_of_max]
        tree={best_attr:{}}
        remaining_attribute_names=[i for i in attribute_names if i!=best_attr]
        for attr_val,data_subset in df.groupby(best_attr):
            subtree=id3(data_subset,target_attribute_name,remaining_attribute_names,default_class)
            tree[best_attr][attr_val]=subtree
        return tree
attribute_names=list(df_tennis.columns)
print("List of Attributes:",attribute_names)
attribute_names.remove('playtennis')
print("Predicting Attribute",attribute_names)
from pprint import pprint
tree=id3(df_tennis,'playtennis',attribute_names)
print("\n\nThe Resultant Decision Tree is:\n")
pprint(tree)
attribute=next(iter(tree))
print("Best Attribute:\n",attribute)
print("Tree Keys:\n",tree[attribute].keys())