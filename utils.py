import os
import time

def log(string, log_path=None, log_name=None, print_info=True, notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    if print_info:
        print(log_string)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = os.path.join(log_path, log_name)
    with open(log_file, 'a+') as f:
        f.write(log_string + '\n')