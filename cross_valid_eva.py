import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import networkx as nx

args = None


def ensure_directory(path: str) -> None:
    """Create path if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)
def w_lev_dis(list1, list2):
    """Calculate a Weighted Levenshtein Distance between two integer lists, where the cost of substitution is
    proportional to the difference between the values."""
    len_list1, len_list2 = len(list1), len(list2)
    dp = [[0] * (len_list2 + 1) for _ in range(len_list1 + 1)]
    total_len=(len_list1+len_list2)/2
    for i in range(len_list1 + 1):
        for j in range(len_list2 + 1):
            if i == 0:
                dp[i][j] = j  # Deletion
            elif j == 0:
                dp[i][j] = i  # Insertion
            elif list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                substitution_cost = abs(list1[i - 1] - list2[j - 1])
                dp[i][j] = min(dp[i - 1][j] + 1,    # Insertion
                               dp[i][j - 1] + 1,    # Deletion
                               dp[i - 1][j - 1] + substitution_cost # Weighted Substitution
                              )
    return round(dp[-1][-1]/total_len,3)
def sem_w_lev_dis(list1, list2):
    """Calculate a Weighted Levenshtein Distance between two integer lists, where the cost of substitution is
    proportional to the difference between the values."""
    len_list1, len_list2 = len(list1), len(list2)
    dp = [[0] * (len_list2 + 1) for _ in range(len_list1 + 1)]
    total_len=(len_list1+len_list2)/2
    for i in range(len_list1 + 1):
        for j in range(len_list2 + 1):
            if i == 0:
                dp[i][j] = j  # Deletion
            elif j == 0:
                dp[i][j] = i  # Insertion
            elif list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
#                print(list1[i-1],list2[j-1])
                substitution_cost = cfg_dis(cfg,list1[i - 1],list2[j - 1])
                dp[i][j] = min(dp[i - 1][j] + 1,    # Insertion
                               dp[i][j - 1] + 1,    # Deletion
                               dp[i - 1][j - 1] + substitution_cost # Weighted Substitution
                              )
    return round(dp[-1][-1]/total_len,3)
def random_list(randlen,totalline):
    return [random.randint(1, totalline) for _ in range(randlen)]
def random_step():
    f_list=[f for f in os.listdir('scanpath') if f.split('.')[0][-3:]=='all']
    column=['token','line','column','semantic','controlnum','feature','complexity','section','stimuli','pid']
    df=pd.DataFrame(columns=column)
    stepsti=[]
    for f in f_list:
        df_f=pd.read_csv("scanpath/"+f)
        df=pd.concat([df,df_f])
        df.reset_index(drop=True, inplace=True)
        for sti in list(set(df['stimuli'].to_list())):
            df_sti=df[df['stimuli']==sti]
            for s in list(set(df_sti['pid'].to_list())):
                stepsti.append(len(df_sti[df_sti['pid']==s]))
    return stepsti
def find_min_distance_sublists(a, b, distance_func):
  min_distance = float('inf')
  closest_pairs = []
  for sublist_a in a:
    current_distance = distance_func(sublist_a, b)
    if current_distance < min_distance:
      min_distance = current_distance
      closest_pairs = [(sublist_a,b)]
    elif current_distance == min_distance:
      closest_pairs.append((sublist_a, b))
  return closest_pairs, min_distance
def eva_aug(t,tname):
  seman="stim_info/"+t+'_sem.csv'
  df_seman=pd.read_csv(seman)
#  cfg = nx.read_gpickle("cfg/"+t[4:]+"_graph.gpickle")
  totalline=df_seman['Line'].to_list()[-1]
  columns=['stim','cross','num','seed','aug','minsimu','dist','pidselect','half_pid']
  df_result=pd.DataFrame(columns=columns)
  f_list=[f for f in os.listdir(f'cross_valid_simu{args.aug[0]}') if (f.split('_')[0]==t and f.split('_')[1]==tname and f.split('_')[3]=='aug'+str(args.aug))]
  df_e=pd.read_csv("all_scanpath_repeat/sti_"+str(t[4:])+"_all.csv")
  pidlist=list(set(df_e['pid'].to_list()))
  chunk_aug=args.aug+'0'
  result_dir = f'cross_valid_result{args.aug[0]}'
  ensure_directory(result_dir)
  chunk_out=os.path.join(result_dir, t+"_"+tname+'_aug'+chunk_aug+'.csv')

#  if not os.path.exists(chunk_out):
#    df_result_chunk=pd.DataFrame(columns=columns)
#    for i in range(abs(int(args.aug))):
#      chunk_f=[f for f in f_list if f.split('_')[4]=='chunk'+str(i+1)][0]
#    print(chunk_f)
#      chunk_select=str(pd.read_csv('cross_valid_simu/'+chunk_f)['pidselect'][0]).split('_')
#      chunk_eva_list=[p for p in pidlist if str(p) not in chunk_select]
#    print(chunk_eva_list)
#      chunk_eva=[]
#      for pid in chunk_eva_list:
#        chunk_eva.append(df_e[df_e['pid']==pid]['line'].to_list())
#    print(chunk_eva)
#      for chunk_sti in chunk_select:
#      print(df_e[df_e['pid']==chunk_sti])
#        chunk_line=df_e[df_e['pid']==int(chunk_sti)]['line'].to_list()
#      print(chunk_line)
#        minpair_chunk, mindist_chunk = find_min_distance_sublists(chunk_eva,chunk_line, w_lev_dis)
#        minpair_sem_chunk, mindist_sem_chunk = find_min_distance_sublists(chunk_eva,chunk_line, sem_w_lev_dis,cfg)
#        df_result_chunk.loc[len(df_result_chunk.index)]=[t,tname,str(i+1),args.seed,args.aug,minpair_chunk,str(mindist_chunk),'_'.join(chunk_select)]
#      print(mindist_chunk)


  try:
    pidselect=str(pd.read_csv(f'cross_valid_simu{args.aug[0]}/'+f_list[0])['pidselect'][0]).split('_')
  except:
    print(f'cross_valid_simu{args.aug[0]}/'+f_list[0])

  for f in f_list:
    df=pd.read_csv(f'cross_valid_simu{args.aug[0]}/'+f)
    linelist=df['Line'].to_list()
    
    pidselect=str(df['pidselect'][0]).split('_')
    halfselect=str(df['half_pid'][0]).split('_')
#    print(pidselect)
#    pidlist_select=[p for p in pidlist if str(p) not in pidselect]
    evalist=[]
#    print(halfselect)
    for pid in halfselect:
      evalist.append(df_e[df_e['pid']==int(pid)]['line'].to_list())
#    print(evalist)
#    print(pidlist)
    if(len(linelist)==0):
      continue
    minpair, mindist = find_min_distance_sublists(evalist,linelist, w_lev_dis)
#    minpair_sem, mindist_sem = find_min_distance_sublists(evalist,linelist, sem_w_lev_dis,cfg)
#  print('dist: '+str(distance)+"  "+str(min(distance)/len(linelist)))
#    mindist=round(mindist/len(linelist),3)
    df_result.loc[len(df_result.index)]=[t,tname,f.split('_')[2],args.seed,args.aug,minpair,str(mindist),'_'.join(pidselect),'_'.join(halfselect)]
  out=os.path.join(result_dir, t+"_"+tname+'_aug'+args.aug+'.csv')
  df_result.to_csv(out,index=False)
def eva_zero(t,tname):
  seman=t+'_sem.csv'
  df_seman=pd.read_csv(seman)
  totalline=df_seman['Line'].to_list()[-1]
#  print(totalline)
#  cfg = nx.read_gpickle("cfg/"+t[4:]+"_graph.gpickle")
#  print(t)
  f_list=[f for f in os.listdir('cross_valid_simu') if (f.split('_')[0]==t and f.split('_')[1]==tname and f.split('_')[3]=='aug0')]
#  print(f_list)
  df_e=pd.read_csv("all_scanpath_repeat/sti_"+str(t[4:])+"_all.csv")
  pidlist=list(set(df_e['pid'].to_list()))
  evalist=[]
  random_steplist=random_step()
  columns=['stim','cross','num','seed','aug','minsimu','minranline','minranall','dist','rand_dist_line','rand_dist_overall']
  df_result=pd.DataFrame(columns=columns)
  for pid in pidlist:
    evalist.append(df_e[df_e['pid']==pid]['line'].to_list())
  for f in f_list:
    df=pd.read_csv(f'cross_valid_simu{args.aug[0]}/'+f)
    linelist=df['Line'].to_list()
    if(len(linelist)==0):
      continue
    randlist_overall=random_list(random.choice(random_steplist),totalline)
    randlist_line=random_list(len(linelist),totalline)
    distance=[]
    randis_line=[]
    randis_overall=[]    
    minpair, mindist = find_min_distance_sublists(evalist,linelist, w_lev_dis)
    minranline, min_ranline_distance = find_min_distance_sublists(evalist, randlist_line,w_lev_dis)            
    minranall, min_ranall_distance = find_min_distance_sublists(evalist,randlist_overall,  w_lev_dis) 
#  print(len(eva))
#    mindist=round(mindist/len(linelist),3)
#    rand_dist_line=round(min_ranline_distance/len(linelist),3)
#    rand_dist_overall=round(min_ranall_distance/len(linelist),3)
    df_result.loc[len(df_result.index)]=[t,f.split('_')[1],f.split('_')[2],args.seed,0,minpair,minranline,minranall,str(mindist),str(min_ranline_distance),str(min_ranall_distance)]
#  print(df_result)
  result_dir = 'cross_valid_result'
  ensure_directory(result_dir)
  out=os.path.join(result_dir, t+'_'+tname+"_aug0.csv")
  df_result.to_csv(out,index=False)

def parse_args():
  parser=argparse.ArgumentParser()
  parser.add_argument('--s_list',required=True)
  parser.add_argument('--t', required=True)
  parser.add_argument('--seed',required=True)
  parser.add_argument('--aug',required=True)
  return parser.parse_args()


def main():
  global args
  args=parse_args()
  slist=args.s_list.split(' ')
  tlist=[args.t]
  seed_value = int(args.seed)
  random.seed(seed_value)
  tname='x'.join([i[4:] for i in tlist])
  if int(args.aug)>0 or int(args.aug)< -2:
    for t in tlist:
      eva_aug(t,tname)
  if int(args.aug)==0:
    for t in tlist:
      eva_zero(t,tname)


if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  main()
