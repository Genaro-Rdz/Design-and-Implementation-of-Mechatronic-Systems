import sys
import time
import setup
import functions

functions.reset()
print("Sistema listo")

while True:
    try:
        cmd = sys.stdin.readline()
        functions.commands(cmd)
    except Exception as e:
        print("Commando no encontrado")
        
    time.sleep(0.05)
