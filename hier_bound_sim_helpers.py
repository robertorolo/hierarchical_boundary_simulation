from itertools import product
from itertools import permutations
from itertools import combinations
import numpy as np
import collections
import pandas as pd
from scipy import spatial
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def ijk_in_n(grid, i, j, k):
    dims = [grid.grid_array[0], grid.grid_array[3], grid.grid_array[6]]
    n = k*dims[0]*dims[1]+j*dims[0]+i
    return n

def marching_cubes(grid, prop):
    print('Marching the cubes...')
    nx, ny, nz = int(grid.grid_array[0]), int(grid.grid_array[3]), int(grid.grid_array[6])
    
    #c_delim = np.ones(len(prop)) * float('nan')
    c_delim = np.zeros(len(prop))
    
    if nz != 1:
        print('3D')
        range_x, range_y, range_z = [l for l in range(0,nx-1)], [l for l in range(0,ny-1)], [l for l in range(0,nz-1)]
        contacts = []       
        for k, j, i in product(range_z, range_y, range_x):       
            cube = np.array([[i,j,k],[i,j+1,k],[i+1,j+1,k],[i+1,j,k],[i,j,k+1],[i,j+1,k+1],[i+1,j+1,k+1],[i+1,j,k+1]])
            indices = [int(ijk_in_n(grid, e[0], e[1], e[2])) for e in cube]
            cats = prop[indices]
            cats = cats[~np.isnan(cats)]
            if np.unique(cats).size > 1:
                contacts.append(np.unique(cats))
                c_delim[indices] = len(np.unique(cats))

    else:
        print('2D')
        range_x, range_y = [l for l in range(0,nx-1)], [l for l in range(0,ny-1)]
        contacts = []        
        for j, i in product(range_y, range_x):       
            cube = np.array([[i,j,0],[i,j+1,0],[i+1,j+1,0],[i+1,j,0]])
            indices = [int(ijk_in_n(grid, e[0], e[1], e[2])) for e in cube]
            cats = prop[indices]
            cats = cats[~np.isnan(cats)]
            if np.unique(cats).size > 1:
                contacts.append(np.unique(cats))
                c_delim[indices] = len(np.unique(cats)) 
    
    return contacts, c_delim
    
def unique_contacts(contacts):
    unique_contacts = []
    for idx, i in enumerate(contacts):
        
        p = list(permutations(i))
        
        # Iterate in the 1st list
        check = False
        for m in unique_contacts: 
      
            # Iterate in the 2nd list 
            for n in p: 
        
                # if there is a match
                if m == n: 
                    check = True 
        
        if check is False:
            unique_contacts.append(tuple(i))
    
    return unique_contacts
    
def contacts_count(unique_contacts, contacts):
    print('Counting contacts...')
    unique_count = {}
    for j in unique_contacts:
    
        p = list(permutations(j))

        count = 0
        for i in p:
            for l in contacts:
                if np.array_equal(l,i):
                    #print(l, i)
                    count=count+1

        unique_count[j] = count
    
    unique_count = dict( sorted(unique_count.items(),
                           key=lambda item: item[1],
                           reverse=True))
    
    return unique_count
    
def sub_contacts(unique_count):
    #not using it yet
    #this is for the cases when triple contacts are more frequent. Is it possible?
    most_freq_contact = list(unique_count.keys())[0]
    siz = len(most_freq_contact)
    s_contacts = {}
    while siz >= 3:
        siz = siz - 1
        comb = combinations(most_freq_contact, siz)

        for i in comb:
            p = list(permutations(i))
                
            #Iterate in the 1st list
            check = False
            for m, c in zip(list(unique_count.keys()), list(unique_count.values())): 

                #Iterate in the 2nd list 
                for n in p: 

                    #if there is a match
                    if m == n: 
                        check = True 
            
                if check == True:
                    s_contacts[m] = c
    
    s_contacts = dict(sorted(s_contacts.items(), key=lambda item: item[1], reverse=True))
    
    return s_contacts
    
def remove_most_frequent(dic):
    check = list(dic.keys())[0]
    for key, val in zip(list(dic.keys()), list(dic.values())):
        for c in check:
            if c in key:
                if key in dic.keys():
                    dic.pop(key)
    return dic
    
def grouping(df, rock, grid, prop):

    df = df.data
    
    maps = {}
    
    contacts, c_delim = marching_cubes(grid, prop)
    u_contacts = unique_contacts(contacts)
    unique_count = contacts_count(u_contacts, contacts)
    print('Contacts count: ', unique_count)
    
    unique_cats = df[rock].unique()
    rest_of_cats = list(unique_cats)
    
    for it in range(len(unique_cats-1)):
        maps['g{}'.format(it+1)] = {}
        if it%2 == 0:
            most_freq_contact = list(unique_count.keys())[0]
            for i in most_freq_contact:
                maps['g{}'.format(it+1)][i] = 1 #if it == 0 else 0
                rest_of_cats.remove(i)
            for j in rest_of_cats:
                maps['g{}'.format(it+1)][j] = 0 #if it == 0 else 1

        else:
            for idx, k in enumerate(most_freq_contact):
                if idx == 0:
                    maps['g{}'.format(it+1)][k] = 1
                else:
                    maps['g{}'.format(it+1)][k] = 0
                    remove_most_frequent(unique_count)
        
            if len(rest_of_cats) == 1:
                print('Done!')
                break
            
    return c_delim, maps
    
def df_groups(maps, df, rock):
    groups = {}
    for i in maps:
        dfg = df.data.copy()
        if i == 'g1':
            dfg[rock] = dfg[rock].map(maps[i])
            groups[i] = dfg
        else:
            cats = maps[i].keys()
            dfg = dfg[dfg[rock].isin(cats)]
            
            dfg[rock] = dfg[rock].map(maps[i])
            groups[i] = dfg
    return groups
    
    
def backflag(x, y, z, prop, grid, reals, output='images/backflag.png'):
    prop = prop.values
    codes = np.unique(prop)
    xg, yg, zg = grid.get_coordinates()
    gpts = np.array([xg, yg, zg]).T
    tree = spatial.KDTree(gpts)
    pts = np.array([x, y, z]).T
    ids = []
    for p in pts:
        i = tree.query(p)[1]
        ids.append(i)
    
    reals_values = [reals[r].values[ids] for r in reals.columns]
    cms = [confusion_matrix(prop, pred) for pred in reals_values]
    sum_ew = np.sum(cms, axis=0)
    final_cm = sum_ew / sum_ew.astype(np.float).sum(axis=1)
    
    plt.figure(figsize=(6,5))
    sns_plot = sns.heatmap(final_cm, annot=True, vmin=0.0, vmax=1.0, fmt='.2f')
    plt.yticks(np.arange(len(codes))+0.5, labels=codes)
    plt.xticks(np.arange(len(codes))+0.5, labels=codes)
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    figure = sns_plot.get_figure()
    figure.savefig(output, dpi=300)
    
def volume_diff(reals, output='images/bars.png'):
    
    cats = np.unique(reals[reals.columns[0]].values)
    conts = []
    for cat in cats:
        cont_cat = []
        for col in reals.columns:
            cont = reals[col].tolist().count(cat)
            cont_cat.append(cont)
        conts.append(cont_cat)
        
    mins, maxs = [min(conts[i])/1000 for i in range(len(cats))], [max(conts[i])/1000 for i in range(len(cats))]
    plt.figure(figsize=(5,5))
    plt.grid()
    plt.bar(cats, mins, label='min', width=0.25)
    plt.bar(np.array(cats) + 0.25, maxs, label='max',  width=0.25)
    plt.legend(fontsize=10)
    plt.ylabel('in K blocks', fontsize=10)
    plt.xlabel('category', fontsize=10)
    plt.xticks(ticks=cats+0.1, labels=[int(cat) for cat in cats])
    plt.savefig(output, dpi=300)
    plt.show()
    
def entropy(reals):
    
    cats = np.unique(reals[reals.columns[0]])
    shanon_entropy = []
    
    for idx, row in reals.iterrows():
        entropy = 0
        a = row.values
        for cat in cats:
            p_cat = np.sum(a==cat)/len(a)
            if p_cat > 0:
                e = - p_cat * np.log(p_cat)
                entropy = entropy + e
        shanon_entropy.append(entropy)
    
    return np.array(shanon_entropy)