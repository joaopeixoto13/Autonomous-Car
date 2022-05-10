# made by: Ines A. Ribeiro - Electronisc Engeneering Uminho - a77258
# software to connect to coppeliaSim simulator (car_treino)
# and control the car's velocity and direction using the keys: w, a, s, d
import math
import sim
import time
import cv2
import numpy as np
import sys
from state_machine import encode_FSM
from signals import detectSignal

# N A O   E S Q U E C E R
# fazer esta linha no coppelia antes de iniciar simulação
# se usar a porta 19997, está sempre aberta
# simRemoteApi.start(19998)
# Define CODEC e cria objecto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',  fourcc, 20.0, (512, 256))    # (width, height)

sim.simxFinish(-1) # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19998, True, True, 5000, 5) # Connect to CoppeliaSim
# inicializar atuadores e sensores
errorL, motorR = sim.simxGetObjectHandle(clientID, "velocityJointR", sim.simx_opmode_oneshot_wait)
errorR, motorL = sim.simxGetObjectHandle(clientID, "velocityJointL", sim.simx_opmode_oneshot_wait)
_, turnR = sim.simxGetObjectHandle(clientID, "directionJointR", sim.simx_opmode_oneshot_wait)
_, turnL = sim.simxGetObjectHandle(clientID, "directionJointL", sim.simx_opmode_oneshot_wait)
errorC, camera = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_oneshot_wait)
errorSC, signalCamera = sim.simxGetObjectHandle(clientID, 'Vision_sensor0', sim.simx_opmode_oneshot_wait)
print(clientID, motorL, motorR, turnL, turnR)

# define valores iniciais
vel_base = 2.5
ang_base = 2.5
vel = 2.5
ang = 0
fps = 0 # so serve para ver quantos frames tenho por segundo
frames = 0
nome = "seq"
tecla = 0
u = 0
count = 0

if clientID != -1:  # verifica que se conseguio ligar a simulação
    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
    time.sleep(1)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
    # se a simulação não tiver começado começa automaticamente
    print("Ligou-se")
    errorC1, res, frame = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_streaming)  # obtem a primeira imagem da camara
    errorC2, res2, frame2 = sim.simxGetVisionSensorImage(clientID, signalCamera, 0, sim.simx_opmode_streaming)  # obtem a primeira imagem da camara
    start = time.time()  # começa a contar o tempo

    while tecla -27:

        tecla = cv2.waitKey(1)
        ang = u

        # atribui valores aos atuadores
        sim.simxSetJointTargetVelocity(clientID, motorR, vel, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, motorL, vel, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, turnR, ang*math.pi/180, sim.simx_opmode_oneshot) # atribuir angulos em radianos
        sim.simxSetJointTargetPosition(clientID, turnL, ang*math.pi/180, sim.simx_opmode_oneshot) # atribuir angulos em radianos
        err, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_buffer) # busca imagem seguinte
        errSC, resolutionSC, signalImage = sim.simxGetVisionSensorImage(clientID, signalCamera, 0, sim.simx_opmode_buffer) # busca imagem seguinte
        #print(resolution)

        if err == sim.simx_return_ok:   
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = cv2.flip(img, 0)

            imgSignal = np.array(signalImage, dtype=np.uint8)
            imgSignal.resize([resolutionSC[1], resolutionSC[0], 3])
            imgSignal = cv2.flip(imgSignal, 0)
            imgSignal = cv2.cvtColor(imgSignal, cv2.COLOR_RGB2BGR)

            vel, u = encode_FSM(img, imgSignal)

            out.write(img)
            frames += 1

            if time.time()-start >= 1:
                start = time.time()
                #print(fps)
                fps = 0
            else:
                fps += 1
        else:
            #print(f"Camera error {count}")
            count += 1
    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
    out.release()
else:
    sys.exit("Nao se ligou")
