
  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tikzplotlib
import numpy as np
import os
from operator import itemgetter

logdir = '/home/akyeast/rubis_ws/log/'
outputdir = '/home/akyeast/rubis_ws/result/'
logfile_name = 'test'
outputfile_name = ''
logfile_path = ''
outputfile_path = ''


logs = {}
    # {
    #     'times': [0.0, 0.01, 0.02],
    #     'topics': ['/ctrl_cmd', '/odom', '/vehicle_cmd_test', '/rubis_log_handler'],
    #     '/ctrl_cmd': {
    #         0.0: {
    #             'ctrl_cmd.cmd.linear_velocity': '10.000000',
    #             'ctrl_cmd.cmd.steering_angle': '0.000000'
    #         },
    #         0.01: {
    #             'ctrl_cmd.cmd.linear_velocity': '10.000000',
    #             'ctrl_cmd.cmd.steering_angle': '0.000000'
    #         },
    #         0.02: {
    #             'ctrl_cmd.cmd.linear_velocity': '10.000000',
    #             'ctrl_cmd.cmd.steering_angle': '0.000000'
    #         }
    #     },
    #     '/odom': {
    #         0.0: {
    #             'odom.twist.twist.linear.x': '0.000000',
    #             'odom.twist.twist.angular.z': '-0.000000'
    #         },
    #         0.01: {
    #             'odom.twist.twist.linear.x': '0.000000',
    #             'odom.twist.twist.angular.z': '-0.000001'
    #         },
    #         0.02: {
    #             'odom.twist.twist.linear.x': '0.000000',
    #             'odom.twist.twist.angular.z': '-0.000001'
    #         }
    #     }, 
    #     '/vehicle_cmd_test': {
    #         0.0: {
    #             'vehicle_cmd_test.ctrl_cmd.linear_acceleration': '0.000000',
    #             'vehicle_cmd_test.ctrl_cmd.steering_angle': '0.000000'
    #         },
    #         0.01: {
    #             'vehicle_cmd_test.ctrl_cmd.linear_acceleration': '0.000000',
    #             'vehicle_cmd_test.ctrl_cmd.steering_angle': '0.000000'
    #         },
    #         0.02: {
    #             'vehicle_cmd_test.ctrl_cmd.linear_acceleration': '1.000000',
    #             'vehicle_cmd_test.ctrl_cmd.steering_angle': '0.000000'
    #         }
    #     },
    #     '/rubis_log_handler': {
    #         0.0: {
    #             'rubis_log_handler.writeon': '0'
    #         },
    #         0.01: {
    #             'rubis_log_handler.writeon': '0'
    #         },
    #         0.02: {
    #             'rubis_log_handler.writeon': '0'
    #         }
    #     }
    # }
def calculate_pid_score(target_vel):
    # for fixed target_velocity exp

    max_accel = 1
    time_start = None
    time_end = None
    time_sat = None
    logging = False
 
    last_log_handler = '0'
    for t in logs['times']:
        log_handler = logs['/rubis_log_handler'][t]['rubis_log_handler.writeon']
        if log_handler == '1' and last_log_handler == '0':
            logged = True
            time_start = t
        if log_handler == '0' and last_log_handler == '1':
            logged = False
            time_end = t

        if logged:
            real_speed = logs['/car_ctrl_output'][t]['car_ctrl_output.real_speed']
            if real_speed >= target_vel*0.95 and real_speed <= target_vel * 1.05:
                time_sat = t

    # milli second
    pid_score = (time_sat - time_start) * 1000 - target_vel / max_accel * 1000 * 0.95

    print(f'time_start: {time_start}')
    print(f'time_sat: {time_sat}')
    print(f'saturation_time: {(time_sat - time_start) * 1000}')
    print(f'pid score: {pid_score}')


            


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
    time = u_ts + '.' + msec
    return float(time)

def parse_log():
    with open(logfile_path, 'r') as f:
        logger_start_time = 0.0
        number_target_topics = 0
        time_first = 0.0
        time_instance = 0.0
        topic_instance = ''

        logs['times'] = []
        logs['topics'] = []

        for line in f:
            # print(line.split('\n')[0])  #line includes \n
            line_wo_nl = line.split('\n')[0]
            attr = line_wo_nl.split(': ')[0]
            data = line_wo_nl.split(': ')[-1]
            
            if attr == '-------------------------------------------------------':
                continue
            
            #parsing log header
            elif attr == 'start_time':
                logger_start_time = float(data)
            elif attr == 'number_target_topics':
                number_target_topics = int(data)
            elif attr == 'topic':
                logs[data] = {}
                logs['topics'].append(data)
            

            #parsing log datas
            elif attr == 'time':
                # print(data)
                time = logtimeconverter(data)
                if time_first == 0.0:
                    time_first = time
                # print(time)
                time_instance = round(time-time_first, 2)
                # print(time_instance)
                logs['times'].append(time_instance)
            elif attr == 'target_topic':
                topic_instance = data
                logs[topic_instance][time_instance] = {}
            else:
                print(attr)
                logs[topic_instance][time_instance][attr] = data
            
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

    # print(logs)

    calculate_pid_score(10)
    

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