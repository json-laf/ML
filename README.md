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
    
    
demo = {
    "proof": {
        "a": [
            "0x0982e17928a3206ed5f7621124b46fa166891384956cc1f05bb94963c81d5550",
            "0x172e11667f418b018da4e0433fa9fc468f7c2ed2b25a76dc85c161d567117034"
        ],
        "b": [
            [
                "0x2ec78845326e89634f70e15f0b173308c2339c4f9db51b8a37581e960d599628",
                "0x09ac99a223fece88a0f9886a2922bc3526712f8f8c901ecc7d2a41454b003fef"
            ],
            [
                "0x11d7cc579da7832026e40fb3a4c7b0ed46c471f73a7d0a439977c83b5c3ff499",
                "0x14c4d22df6f1de0e4bc616036e3c29ea042019859b3afd6374a37addefd37ba7"
            ]
        ],
        "c": [
            "0x259bd67b95250264e0da1e5f8dccdb417bdaa4576df78f7c8d98719de0d9e6f3",
            "0x290db70cb8b6f6a50772d998068aa519c6e55da99d6a917a6b0654fd600965f8"
        ]
    },
    "inputs": [
        "0x0000000000000000000000000000000000000000000000000000000000000004"
    ]
}
