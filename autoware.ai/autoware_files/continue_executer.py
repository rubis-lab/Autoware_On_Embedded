import sys
import signal
import subprocess

stop_requested = False
command_pid = -1

def SIGINT_handler(signum, frame):
    global stop_requested
    stop_requested = True
    if command_pid >= 0:
        subprocess.Popen.kill(command_pid)
    print('Exit with Ctrl+C')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Insufficient arguments; Usage: python3 continue_executer.py $(cmd)")
        exit(1)

    signal.signal(signal.SIGINT, SIGINT_handler)
    
    cmd = sys.argv[1:]

    while not stop_requested:
        pid = subprocess.Popen(cmd)
        pid.wait(timeout=None)