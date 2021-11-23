# ML

import psutil
import time

start = time.perf_counter()
doc = open('资源消耗.txt','w')
if __name__ == '__main__':
    cpu_ls = []
    mem_ls = []
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            mem_percent = psutil.virtual_memory().percent
            cpu_ls.append(round(cpu_percent/100, 3))
            mem_ls.append(round(mem_percent/100, 3))
            print(cpu_percent, mem_percent)
            time.sleep(1)
        except Exception as e:
            print(e)
        # finally:
        end = time.perf_counter()
        if int(end - start) >= 60:
            print(end - start)
            break
    print(f'CPU占用率:{cpu_ls},\n内存占用率:{mem_ls}', file=doc)
