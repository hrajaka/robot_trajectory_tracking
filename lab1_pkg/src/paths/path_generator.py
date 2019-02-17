import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

a = 5/40
nb_subdiv = 10

def acc(t, t_tot):
    if t < t_tot/nb_subdiv:
        return a * nb_subdiv / t_tot * t
    elif t < 2*t_tot/nb_subdiv:
        return - a * nb_subdiv / t_tot * (t-t_tot/nb_subdiv) + a
    elif t < 8*t_tot/nb_subdiv:
        return 0
    elif t < 9*t_tot/nb_subdiv:
        return - a * nb_subdiv / t_tot * (t-8*t_tot/nb_subdiv)
    elif t < t_tot:
        return a * nb_subdiv / t_tot * (t-9*t_tot/nb_subdiv) - a
    else:
        return 0


def vel(t, t_tot):
    if t <= t_tot/nb_subdiv:
        return quad(acc, 0, t, args=(t_tot,))[0]
    elif t <= 2*t_tot/nb_subdiv:
        return quad(acc, t_tot/nb_subdiv, t, args=(t_tot,))[0] + vel(t_tot/nb_subdiv, t_tot)
    elif t <= 8*t_tot/nb_subdiv:
        return quad(acc, 2*t_tot/nb_subdiv, t, args=(t_tot,))[0] + vel(2*t_tot/nb_subdiv, t_tot)
    elif t <= 9*t_tot/nb_subdiv:
        return quad(acc, 8*t_tot/nb_subdiv, t, args=(t_tot,))[0] + vel(8*t_tot/nb_subdiv, t_tot)
    elif t <= t_tot:
        return quad(acc, 9*t_tot/nb_subdiv, t, args=(t_tot,))[0] + vel(9*t_tot/nb_subdiv, t_tot)
    else:
        return 0


def pos(t, t_tot):
    if t <= t_tot/nb_subdiv:
        return quad(vel, 0, t, args=(t_tot,))[0]
    elif t <= 2*t_tot/nb_subdiv:
        return quad(vel, t_tot/nb_subdiv, t, args=(t_tot,))[0] + pos(t_tot/nb_subdiv, t_tot)
    elif t <= 8*t_tot/nb_subdiv:
        return quad(vel, 2*t_tot/nb_subdiv, t, args=(t_tot,))[0] + pos(2*t_tot/nb_subdiv, t_tot)
    elif t <= 9*t_tot/nb_subdiv:
        return quad(vel, 8*t_tot/nb_subdiv, t, args=(t_tot,))[0] + pos(8*t_tot/nb_subdiv, t_tot)
    elif t <= t_tot:
        return quad(vel, 9*t_tot/nb_subdiv, t, args=(t_tot,))[0] + pos(9*t_tot/nb_subdiv, t_tot)
    else:
        return 0

if __name__ == '__main__':

    t_tot = 10
    time = np.linspace(0, t_tot, 50)
    for t in time:
        plt.scatter(t, acc(t, t_tot), c='b', marker='.')
        plt.scatter(t, vel(t, t_tot), c='r', marker='.')
        plt.scatter(t, pos(t, t_tot), c='g', marker='.')
    plt.grid()
    plt.show()


















#
