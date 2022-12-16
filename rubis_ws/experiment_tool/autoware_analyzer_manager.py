import os
import sys, getopt
from autoware_analyzer import *

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"hc:ep",["help","center=","e2e", "plot"])
   except getopt.GetoptError:
      print("python3 autoware_analyzer_manager.py -h/--help")
      sys.exit(1)
   for opt, arg in opts:
      if opt in ("-h", "--help") :
         print("usage: python3 autoware_analyzer_manager.py [option] [arg]")
         print("Options and arguments")
         print("-h                 : print this help message and exit (also --help)")
         print("-c {file_name.csv} : collect center offset data and save it to file_name.csv (also --center)")
         print("-e                 : get log from board and calculate e2e (also --e2e)")
         print("-p                 : plot the center off and e2e (also --plot)")
         sys.exit(2)
      elif opt in ("-c", "--center"):
         # update yaml
         data=[]
         with open('autoware_analyzer.yaml') as f:
            data = yaml.load(f, yaml.FullLoader)
            if not os.path.isdir('./data/'):
               command = 'mkdir ./data'
               os.system(command)
            if not os.path.isdir('./data/' + argv[1]):
               command = "mkdir ./data/" + argv[1]
               os.system(command)
            if not os.path.isdir('./data/' + argv[1] + '/autoware_log'):
               command = "mkdir ./data/" + argv[1] + "/autoware_log"
               os.system(command)
            data['center_offset_path'] = './data/' + argv[1] + "/" + argv[1] + "_center_offset.csv"
            data['e2e_response_time_path'] = './data/' + argv[1] + "/" + argv[1] + "_e2e.csv"
            data['plot_path'] = './data/' + argv[1] + "/" + argv[1] + "_plot.png"
            data['node_paths']['lidar_republishr'] = './data/' + argv[1] + '/autoware_log/lidar_republisher.csv'
            data['node_paths']['twist_gate'] = './data/' + argv[1] + '/autoware_log/twist_gate.csv'
         with open('autoware_analyzer.yaml', 'w') as f:
            yaml.dump(data,f,default_flow_style=True)
         # collect center offset
         center_off_ndt(argv[1]+'.csv')
         
      elif opt in ("-e", "--e2e"):
         with open('autoware_analyzer.yaml') as f:
            data = yaml.load(f, yaml.FullLoader)
            command = "cp ~/Documents/profiling/response_time/* ./data/" + data['center_offset_path'].split('/')[2] + "/autoware_log"
            os.system(command)
         calculate_e2e_response_time()
      elif opt in ("-p", "--plot"):
         plot_e2e_and_center_offset_by_instance()
   if len(argv) == 0:
      print("python3 autoware_analyzer_manager.py -h/--help")

if __name__ == "__main__":
   main(sys.argv[1:])