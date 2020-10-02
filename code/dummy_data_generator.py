import numpy as np
import random

MAX_SEQ_LEN=20

def generate_dummy_data():
    # Dummy data:
    #  x_t = ax_{t-1} + \epsilon_x
    #  y_t = by_{t-1} + cx_{t} + \epsilon_y
    a_1 = 1.05; a_2 = 1.1; a_3 = 0.9
    b_1 = 0.90; b_2 = 0.90
    c_1 = 0.49; c_2 = 0.4
    x = []
    y = []

    for i in range(100):
        seq_length = random.randint(10, MAX_SEQ_LEN)
        x_i = [[1, 1, 1]]
        y_i = [[1, 1]]
        for t in range(MAX_SEQ_LEN)[1:]:
            if t < seq_length:
                x_t = [
                    a_1*x_i[t-1][0], #+ random.gauss(0, 0.05),
                    a_2*x_i[t-1][1], #+ random.gauss(0, 0.05),
                    a_3*x_i[t-1][2], #+ random.gauss(0, 0.05),
                ]
                x_i.append(x_t)

                y_t = [
                    b_1*y_i[t-1][0] + c_1*x_i[t][0], #+ random.gauss(0, 0.05),
                    b_2*y_i[t-1][1] + c_2*x_i[t][1], #+ random.gauss(0, 0.05),
                ]
                y_i.append(y_t)
            else:
                x_i.append([-1, -1, -1])
                y_i.append([-1, -1])
        x.append(x_i)
        y.append(y_i)
    x = np.array(x)
    y = np.array(y)
    y_target = np.roll(y, -1, axis=1)
    return x, y, y_target, {
        'DATASET_SIZE': 1000,
        'MIN_SEQ_LEN': 10,
        'MAX_SEQ_LEN': MAX_SEQ_LEN,
        'MASK_VALUE': -1,
        'X_DIM': 3,
        'Y_DIM': 2,
    }
