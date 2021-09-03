# tt
# 2021.9.2
# Simple bayes filter demos

import numpy as np
import random as rd

def getUncertainty(possibility) -> bool:
    '''
    :param possibility: the possibility that this function returns true
    :return:
    '''
    return rd.randint(1, 100) <= int(possibility * 100)

def DiscreteBayesFilterDemo():
    '''
    Suppose a robot walking around a discrete labeled area: 0 - 9.
    There are doors at position 0, 1 and 8, which may cause the sensor to return 1.
    The sensor has an accuracy at 80%
    Each time the robot receives a command:
    80% possible to move 1 step (index increasing), 10% possible to stay still and 10% possible to move 2 steps.
    Initial position unknown.
    :return:
    '''
    # Ground truth position
    pos_true = 3
    ROOM_SIZE = 10
    DOORS = [0, 1, 8]
    SENSOR_ACC = 0.8

    # Receive current sensor data at 80% accuracy
    def getSensor():
        nonlocal pos_true
        flag = getUncertainty(SENSOR_ACC)
        if pos_true in DOORS:
            return flag
        return not flag

    # Send command at 80% 10% 10% possibilities
    def sendCommand():
        nonlocal pos_true
        if getUncertainty(0.8):
            pos_true = (pos_true + 1) % ROOM_SIZE
        else:
            if getUncertainty(0.5): # 10% to 10%
                pos_true = (pos_true + 2) % ROOM_SIZE


    # Initial prior possibility
    pos_est = np.array([1.0 / ROOM_SIZE] * ROOM_SIZE)

    # Begin simulation
    for i in range(200):
        # 1. Predict
        if i == 1:
            # Initial prior possibility, we dont have any information, so we guess
            pos_est = np.array([1.0 / ROOM_SIZE] * ROOM_SIZE)
        else:
            # We use latest prediction and current input to predict a prior possibility
            sendCommand()
            pos_move_1 = np.concatenate((pos_est[-1:], pos_est[0:-1]))
            pos_move_2 = np.concatenate((pos_est[-2:], pos_est[0:-2]))
            pos_est = pos_est * 0.1 + pos_move_1 * 0.8 + pos_move_2 * 0.1
            pos_est /= pos_est.sum() # Now we have a prior possibility based on physical model and our input
        # 2. Update the prior possibility
        isSensor = getSensor() # Receive a new observation
        door_likelihood = [1.0] * ROOM_SIZE
        for i in range(ROOM_SIZE):
            if i in DOORS and isSensor or i not in DOORS and not isSensor:
                door_likelihood[i] *= 0.8 / (1.0 - 0.8)
        door_likelihood = np.array(door_likelihood)
        pos_est = pos_est * door_likelihood # Get posterior possibility
        pos_est /= pos_est.sum() # Normalization

    print(f"Final true pos: {pos_true}")
    print(f"Estimated pos distribution: {pos_est}")


if __name__ == "__main__":
    DiscreteBayesFilterDemo()










