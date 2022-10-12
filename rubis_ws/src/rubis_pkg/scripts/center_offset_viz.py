import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os.path import exists

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--file', '-f', type=str, required=True)
  parser.add_argument('--time', '-t', type=int, default=40, help='plot until this time')
  args = parser.parse_args()

  ### Center offset
  plt.rcParams["figure.figsize"] = (7,5)
  plt.rcParams["font.size"] = 11.5

  if not exists(args.file):
    print('csv not exists!')
    exit(1)

  csv_data = pd.read_csv(args.file)

  fig, ax = plt.subplots()

  plt.axhline(0, color='gray', lw=1, linestyle=(0, (1, 5)))

  plt.axhline(6, color='black', lw=2)
  plt.axhline(-6, color='black', lw=2)

  time_index = min(args.time * 10, len(csv_data) // 10 * 10)

  ax.plot([t/10 for t in range(time_index)], csv_data['center_offset'][0:time_index], color='gray', linestyle='-', label='Center Offset')

  ax.set_ylabel('Center Offset (m)')
  ax.set_xlabel('Time (s)')

  ax.set_ylim(-20, 20)

  plt.legend()
#   fig.savefig(res_dir + "/fig15a.png")
  plt.show()
