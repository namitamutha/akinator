# =============================================================================
#                                 import libraries
# =============================================================================

import numpy as np
import pandas as pd
eps = np.finfo(float).eps
import pprint; #pretty print
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix 

# =============================================================================
#                                 function definitions
# =============================================================================

#entropy of dataset
def find_entropy(df):
    
    Class = df.keys()[-1]         #Teacher
    entropy = 0                   #initialization
    values = df[Class].unique()   #names of teachers eg:anagha, suraj...
    
    for value in values:          #for each teacher
        fraction = df[Class].value_counts()[value]/len(df[Class])  #
        entropy += -fraction*np.log2(fraction)                     #
    return entropy
  
#entropy of individual atrribute 
def find_entropy_attribute(df,attribute):  #attr = qualification, etc.
    
  Class = df.keys()[-1]  #Teacher 
  target_variables = df[Class].unique()   #names of teachers eg:anagha, suraj...
  variables = df[attribute].unique()      #if gender then unique means female , male
  entropy2 = 0
  
  for variable in variables:                #for unique values in gender
      entropy = 0                           #initialization
      
      for target_variable in target_variables: #teacher in teachers
          num = len(df[attribute][df[attribute]==variable][df[Class] == target_variable]) #if gender and it is female and if target var = anagha is true then count such instances
          den = len(df[attribute][df[attribute]==variable]) # if gender and if it is female count such instances
          fraction = num/(den+eps) #eps for zero denom
          entropy += -fraction*np.log2(fraction+eps) #entropy of gender = female
          
      fraction2 = den/len(df) # (#female/#total-instances ) similarly male for all above too
      entropy2 += fraction2*entropy # entropy of gender
      
  return entropy2


def find_winner(df): #finding best attribut for split max IG (least entropy for attr)
    
    IG = []
    dataset_entropy=find_entropy(df) #datset entropy function call
    
    for key in df.keys()[:-1]: #all attr...columns except last ie. teachers
        IG.append(dataset_entropy-find_entropy_attribute(df,key)) #dataset entropy-attr entropy for all entropies
    
    return df.keys()[:-1][np.argmax(IG)] #return single attr having max ig
  
  
def get_subtable(df, node,value):#node = gender, value = female
    
    return df[df[node] == value] #return subtable of only rows having gender=female 


def buildTree(df,tree=None): #building decision tree
    
    node = find_winner(df) #got best-split attr
    
    attValue = np.unique(df[node]) #forming np array of unique values in selected attr : gender has female and male..
    
    if tree is None: #building tree and subtree and subtree..so on          
        tree={}
        tree[node] = {}
    
    for value in attValue: #value=male or female
        
        subtable = get_subtable(df,node,value) #extract rows having gender=female 
        clValue = np.unique(subtable['Teacher'])  #find unique teachers within subtable  
           
        if len(clValue)==1: #if only single teacher then pure leaf and no need to split ie. leaf node
            tree[node][value] = clValue[0]   # tree[gender][female] = suchitra ->leaf node value                                                  
        else:        #else recursive call to find pure
            tree[node][value] = buildTree(subtable) #build tree from the subtable formed
                 # if tree[gender][value] -> suchitra,anagha,... not pur .. so search for best split  
                 
    return tree

def predict(inst,tree):
    
    for nodes in tree.keys():    # tree.keys=subject,gender,...  not teacher    
        value = inst[nodes]      # value = inst[gender] -> value = female ..suppose
        tree = tree[nodes][value]  # tree= tree[gender][female] ..traverse acc to attr in our instance
     
        prediction = 0  #initialization of prediction
            
        if type(tree) is dict:  #if node is a dict further on then keep traversing
            prediction = predict(inst, tree) #recursive
        else:
            prediction = tree  #predicted teacher and she's not dictionary
            break              #break from loop              
        
    return prediction          #return this prediction

    
# =============================================================================
#                                     main function
# =============================================================================

#loading dataset
df = pd.read_csv("C:/Users/Reena Robert/Downloads/dataset2.csv")
tdf=pd.read_csv("C:/Users/Reena Robert/Downloads/test.csv")

#building decision tree
tree=buildTree(df)
pprint.pprint(tree)
y=tdf['Teacher']
x=tdf.drop('Teacher',axis=1)
queries=x.to_dict(orient="records")
p=pd.DataFrame(columns=["predicted"])

for i in range (len(x)):
    p.loc[i,"predicted"]=predict(queries[i],tree)
    
print("---------------------------------------------------------------------------------------")   
print("\nAccuracy is:",(np.sum(p["predicted"]==tdf["Teacher"])/len(x))*100)
print('\nAccuracy Score :',accuracy_score(y, p) )
print('\nClassification Report :\n',classification_report(y, p) )
print('\nConfusion Matrix :\n',confusion_matrix(y, p) )
print("\n---------------------------------------------------------------------------------------") 
