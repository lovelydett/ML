# tt
# xyt@bupt.cn
# 2021.9.2
# Simple bayes filter demos (discrete and continuous cases)

import numpy as np
import random as rd

def getUncertainty(possibility) -> bool:
    '''
    :param possibility: the possibility that this function returns true
    :return:
    '''
    return rd.randint(1, 100) <= int(possibility * 100)

def getLikelihoodForNormalDistribution(samples: np.ndarray) -> (float, float):
    '''
    :param samples: data
    :return: expectation and standard deviation
    '''
    miu = samples.sum() / samples.shape[0]
    sigma = np.sqrt(samples.var())
    return miu, sigma

def mulNormalDistribution(miu1, sigma1, miu2, sigma2) -> (float, float):
    s1 = sigma1 * sigma1
    s2 = sigma2 * sigma2

    # Scale factor
    # Todo: Why this not work?
    A = np.exp(-1 * (miu1 - miu2) * (miu1 - miu2) / (2 * (s1 + s2))) / np.sqrt(2 * np.pi * (s1 + s2))
    A = 1
    # new expectation
    miu = A * ((miu1 * s2) + (miu2 * s1)) / (s1 + s2)
    sigma = A * np.sqrt((s1 * s2) / (s1 + s2))

    return miu, sigma


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
        if i == 0:
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


def ContinuousBayesFilterDemo():
    '''
    Suppose a burning engine, the temperature inside t(k) at a specific time spot k is not directly measurable
    as a hidden state, but the sensor outside can give an observation y(k) satisfying y(k) = t(k) + Q1, Q1 ~ N(0, sigma1^2).
    Each time we inject fuels, making the x(k) = x(k - 1) + Q2, Q2 ~ N(A, sigma2^2), this is the physical model.
    Initial temperature is unknown, so we have no initial prior knowledge
    :return:
    '''

    # Ground truth
    t_true = 2000.0
    A = 5
    sigma1, sigma2 = 2, 3

    t_est = np.array([0.0, 1.0]) # [expectation, variance]

    def addFuel():
        nonlocal t_true
        q1 = np.random.normal(A, sigma2)
        t_true += q1

    def observe() -> float:
        nonlocal t_true
        q2 = np.random.uniform(0, sigma1)
        return t_true + q2

    # Begin simulation
    for i in range(10):
        # 1. Predict
        if i == 0:
            # We dont have initial information, use observation as first prior knowledge
            y = observe()
            t_est = np.array([y, sigma1 * sigma1])
        else:
            # Use physical model to predict tk
            addFuel()
            t_est += np.array([A, sigma2 * sigma2])

        # 2. Update with observation
        sample_time = 20 # We have to get enough observations to compute a trustable max likelihood
        observes = [observe() for _ in range(sample_time)]
        observes = np.array(observes)
        miu, sigma = getLikelihoodForNormalDistribution(observes)
        miu, sigma = mulNormalDistribution(miu, sigma, t_est[0], t_est[1]) # Likelihood * prior = posterior
        t_est = np.array([miu, sigma])

        print(f"Iteration {i + 1}: actual temp = {t_true}, \t\testimated temp = {t_est[0]}")

    print(f"Final temperature: {t_true}")
    print(f"Estimated temperature: {t_est}")

if __name__ == "__main__":
    ContinuousBayesFilterDemo()









