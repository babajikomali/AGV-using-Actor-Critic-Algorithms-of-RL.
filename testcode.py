import multiprocessing
from environment import Environment, Path
from Differential_Car import Car
import multiprocessing
import random

actions = {
    0: 'left_increase',
    1: 'left_decrease',
    2: 'right_increase',
    3: 'right_decrease',
    4: 'none',
}
def foo():
    path1 = Path()
    env = Environment(path1)
    obs = env.reset()
    action = random.choice([0, 1, 2, 3, 4])
    obs = env.step(actions[action])
    print(obs)
if __name__ == '__main__':
    p1 = multiprocessing.Process(target=foo)
    p2 = multiprocessing.Process(target=foo)
    p3 = multiprocessing.Process(target=foo)
    p4 = multiprocessing.Process(target=foo)
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    print("Done!")
