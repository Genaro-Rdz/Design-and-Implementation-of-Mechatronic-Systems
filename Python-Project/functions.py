from machine import Pin
import setup

output1 = Pin(setup.PIN_OUT1, Pin.OUT)
output2 = Pin(setup.PIN_OUT2, Pin.OUT)
init = False

def commands(cmd):
    global init
    cmd = cmd.strip().upper()
    if cmd == "A":
        print(init)
    elif cmd == "*INIT":
        print("Inicializado")
        init = True
    elif cmd == "*RST":
        reset()
    elif cmd == "*IDN?":
        print(f"{setup.FABRICANTE},{setup.MODELO},{setup.SERIAL},{setup.VERSION}")
    elif cmd == "OUT1 ON":
        if not init:
            print("ERROR: Sistema no inicializado, manda *INIT primero")
        else:
            output1.on()
    elif cmd == "OUT1 OFF":
        if not init:
            print("ERROR: Sistema no inicializado, manda *INIT primero")
        else:
            output1.off()
    elif cmd == "OUT2 ON":
        if not init:
            print("ERROR: Sistema no inicializado, manda *INIT primero")
        else:
            output2.on()
    elif cmd == "OUT2 OFF":
        if not init:
            print("ERROR: Sistema no inicializado, manda *INIT primero")
        else:
            output2.off()
            
def reset():
    global init
    init = False
    output1.off()
    output2.off()