
  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tikzplotlib
import numpy as np
import os
from operator import itemgetter

logdir = '/home/akyeast/rubis_ws/log/'
outputdir = '/home/akyeast/rubis_ws/result/'
logfile_name = ''
outputfile_name = ''
logfile_path = ''
outputfile_path = ''

datas = {}

def find_recentlog():
    recentlog = ''
    for f in os.listdir(logdir):
        if f.split('.')[-1] != 'log':
            continue
        tempname = f.split('.log')[0]
        if tempname[-1] == '.swp':
            continue
        if tempname >= recentlog:
            recentlog = tempname
        else:
            recentlog = recentlog
    return recentlog

def logtimeconverter(logtime):
    u_ts = logtime.split('-')[0]   #unix timestamp, sec
    msec = logtime.split('-')[-1]
    time = u_ts + '.' + msec * 100
    return float(time)

def parse_log():
    with open(logfile_path, 'r') as f:
        parsing_status = None # None, divider, time, topicdef, topic
        for line in f:
            # print(line.split('\n')[0])  #line includes \n
            line_wo_nl = line.split('\n')[0]
            if line_wo_nl.split(':')[0] == '-------------------------------------------------------':
                parsing_status = 'divider'
            elif line_wo_nl.split(':')[0] == 'time':
                parsing_status = 'time'
                line_ = line_wo_nl.split(': ')[-1]
            elif line_wo_nl.split(':')[0] == 'topicdef':
                parsing_status = 'topicdef'
                line_data = line_wo_nl.split(': ')[-1]
            else:
                parsing_status = 'topic'
                line_data = line_wo_nl.split(': ')[-1]
            
            # if parsing_status = 'time':
            #     logtime
            
                # print('there is divider')
            
    return

# def draw_barchart(ntask):
#     strategy = ['Ours', 'Single', 'Max', 'Random']
    
#     # normalize
#     ntask_normalized = []
#     local_maximum = 0
#     for st in ntask:
#         if st >= local_maximum:
#             local_maximum = st
    
    
#     for idx in range(len(ntask)):
#         ntask_normalized.append(ntask[idx]/local_maximum)

#     ntask_percent = []

#     for idx in range(len(ntask_normalized)):
#         ntask_minus = round((ntask_normalized[idx]-1) * 100, 1)
#         ntask_percent.append(ntask_minus)
    
#     # default bar chart
#     plt.figure(figsize=(10, 10))
#     xtick_label_position = list(range(len(strategy)))
#     plt.xticks(xtick_label_position, strategy, fontsize=20)
#     yticks_normalized = [0, 0.5, 1.0, 1.5]
#     ytick_label_position = [0, 0.5, 1.0, 1.5]
#     plt.yticks(ytick_label_position, yticks_normalized, fontsize=20)
#     plt.ylim(0.0, 1.2)
#     bar_component = plt.bar(xtick_label_position, ntask_normalized, edgecolor='black', color = 'white',  width = 0.3, )
#     plt.xlabel('Strategy', fontsize=25)
#     plt.ylabel('# of Schedulable Tasks (Normalized)', fontsize=25)

#     # text upper bar
#     bar_index = 0
#     for rect in bar_component:
#         if ntask_percent[bar_index] != 0.0:
#             height = rect.get_height()
#             plt.text(rect.get_x() + rect.get_width() / 2.0, height+0.02, f'{ntask_percent[bar_index]}%', ha='center', va='bottom', fontsize=20)
#         bar_index += 1

#     # horizontal line
#     plt.axhline(y=1.0, color = 'black', linestyle = '--')

#     plt.savefig(output_path)



if __name__=="__main__":
    
    recentlog = find_recentlog()
    # print(recentlog)

    if logfile_name == '':
        logfile_name = recentlog
    logfile_path = logdir + logfile_name + '.log'
   
    if outputfile_name == '':
        outputfile_name = logfile_name
    outputfile_path = outputdir + outputfile_name + '.pdf'
    
    print(f'parsing\n{logfile_path}\n{outputfile_path}')
    parse_log()

    exit()

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