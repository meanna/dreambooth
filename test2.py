import subprocess, os
command = ['accelerate', 'launch', 'test2.py',]

subprocess.Popen(command, cwd=os.getcwd())
import sys
sys.exit()