
  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tikzplotlib
import numpy as np
import os
from operator import itemgetter

logdir = '/home/akyeast/rubis_ws/log/'
outputdir = 'home/akyeast/rubis_ws/log'
logfile_path = '/home/akyeast/final_1to10.log'
deadline_policy = 0.5
output_path = '/home/rtss/plot.pdf'

def listsort(list_unsorted, idx):
    list_sorted = sorted(list_unsorted, key=itemgetter(idx))
    return list_sorted

# final.log to list
def parse_log(path):
    ret = []
    with open(path, 'r') as f:
        sorted_f = sorted(f)
        for line in sorted_f:
            name_exp = line.split('\t')[0]
            num_line = int(line.split('\t')[1])
            num_deadline = int(line.split('\t')[2])
            num_runtimezero = int(line.split('\t')[3])

            expnum = int(name_exp.split('_')[1].split('exp')[-1])
            tasknum = int(name_exp.split('_')[2].split('task')[-1])
            strategy = name_exp.split('_')[3]

            newlist = []
            newlist.append(expnum)
            newlist.append(tasknum)
            newlist.append(strategy)
            newlist.append(num_line)
            newlist.append(num_deadline)
            newlist.append(num_runtimezero)

            ret.append(newlist)

    # exp: 0, task: 1, strategy: 2, number of lines = 3, number of deadlinemiss = 4, number of runtimeerror = 5
    # print(ret)
    return ret

def expsort (log_parsed):
    st_single = []
    st_max = []
    st_rand = []
    st_ours = []
    exp_single = {}
    exp_max = {}
    exp_rand = {}
    exp_ours = {}

    for ele in log_parsed:
        # print(ele)
        # ignore runtime error
        if ele[5] != 0:
            continue
        
        if ele[2] == 'single':
            st_single.append(ele)
        elif ele[2] == 'max':
            st_max.append(ele)
        elif ele[2] == 'rand':
            st_rand.append(ele)
        elif ele[2] == 'ours':
            st_ours.append(ele)
        else:
            continue
    
    templist = []
    for ele_single in listsort(st_single, 0):
        if not ele_single[0] in exp_single:
            exp_single[ele_single[0]] = []
        exp_single[ele_single[0]].append(ele_single)
    
    for expnum in exp_single:
        templist = listsort(exp_single[expnum], 1)
        exp_single[expnum] = templist
    
    for ele_max in listsort(st_max, 0):
        if not ele_max[0] in exp_max:
            exp_max[ele_max[0]] = []
        exp_max[ele_max[0]].append(ele_max)
    
    for expnum in exp_max:
        templist = listsort(exp_max[expnum], 1)
        exp_max[expnum] = templist

    for ele_rand in listsort(st_rand, 0):
        if not ele_rand[0] in exp_rand:
            exp_rand[ele_rand[0]] = []
        exp_rand[ele_rand[0]].append(ele_rand)
    
    for expnum in exp_rand:
        templist = listsort(exp_rand[expnum], 1)
        exp_rand[expnum] = templist

    for ele_ours in listsort(st_ours, 0):
        if not ele_ours[0] in exp_ours:
            exp_ours[ele_ours[0]] = []
        exp_ours[ele_ours[0]].append(ele_ours)
    
    for expnum in exp_ours:
        templist = listsort(exp_ours[expnum], 1)
        exp_ours[expnum] = templist
    
    return exp_single, exp_max, exp_rand, exp_ours

    
def mean_ntask_possible(exps):
    # exp: 0, task: 1, strategy: 2, number of lines = 3, number of deadlinemiss = 4, number of runtimeerror = 5
    sum_ntask = 0
    for expnum in exps:
        local_max_ntask = 0
        
        for ele in exps[expnum]:
            deadline_miss_ratio = round(ele[4]/ele[3]*100, 4)
            if deadline_miss_ratio > deadline_policy:
                break
            local_max_ntask = ele[1]

        # print(f"local_max_ntask = {local_max_ntask}")
        sum_ntask += local_max_ntask
    
    mean_ntask = sum_ntask / len(exps)
    return mean_ntask
    
def draw_barchart(ntask):
    strategy = ['Ours', 'Single', 'Max', 'Random']
    
    # normalize
    ntask_normalized = []
    local_maximum = 0
    for st in ntask:
        if st >= local_maximum:
            local_maximum = st
    
    
    for idx in range(len(ntask)):
        ntask_normalized.append(ntask[idx]/local_maximum)

    ntask_percent = []

    for idx in range(len(ntask_normalized)):
        ntask_minus = round((ntask_normalized[idx]-1) * 100, 1)
        ntask_percent.append(ntask_minus)
    
    # default bar chart
    plt.figure(figsize=(10, 10))
    xtick_label_position = list(range(len(strategy)))
    plt.xticks(xtick_label_position, strategy, fontsize=20)
    yticks_normalized = [0, 0.5, 1.0, 1.5]
    ytick_label_position = [0, 0.5, 1.0, 1.5]
    plt.yticks(ytick_label_position, yticks_normalized, fontsize=20)
    plt.ylim(0.0, 1.2)
    bar_component = plt.bar(xtick_label_position, ntask_normalized, edgecolor='black', color = 'white',  width = 0.3, )
    plt.xlabel('Strategy', fontsize=25)
    plt.ylabel('# of Schedulable Tasks (Normalized)', fontsize=25)

    # text upper bar
    bar_index = 0
    for rect in bar_component:
        if ntask_percent[bar_index] != 0.0:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height+0.02, f'{ntask_percent[bar_index]}%', ha='center', va='bottom', fontsize=20)
        bar_index += 1

    # horizontal line
    plt.axhline(y=1.0, color = 'black', linestyle = '--')

    plt.savefig(output_path)



if __name__=="__main__":
    
    # parse_log
    log_parsed = parse_log(logfile_path)

    # strategy list / sort by exp num
    exps_single, exps_max, exps_rand, exps_ours = expsort(log_parsed)

    # maximum number of schedulable tasks
    ntask_single = mean_ntask_possible(exps_single)
    ntask_max = mean_ntask_possible(exps_max)
    ntask_rand = mean_ntask_possible(exps_rand)
    ntask_ours = mean_ntask_possible(exps_ours)

    # debug print
    print(f'deadline ratio < {deadline_policy}%')
    print(f'single: {ntask_single}, max: {ntask_max}, rand: {ntask_rand}, ours: {ntask_ours}')

    # draw_chart
    ntask = [ntask_ours, ntask_single, ntask_max, ntask_rand]
    draw_barchart(ntask)


    exit()