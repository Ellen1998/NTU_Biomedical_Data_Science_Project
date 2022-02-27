from ast import Global
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.config import GLOABAL, MAX_EPOCHS

if __name__ == '__main__':
    
    
    Ve_list, Vs_list, Ves_list, Vp_list = [], [], [], []    #List of substance concentration at each time
    # initialize the param
    Ve, Vs, Ves, Vp = GLOABAL.Ve, GLOABAL.Vs, GLOABAL.Ves, GLOABAL.Vp
    
    for i in tqdm(range(MAX_EPOCHS)):

        # record the value
        Ve_list.append(Ve)
        Vs_list.append(Vs)
        Ves_list.append(Ves)
        Vp_list.append(Vp)

        # update the param
        Ve = Ve - GLOABAL.k1*GLOABAL.dt*Ve*Vs + GLOABAL.k2*GLOABAL.dt*Ves + GLOABAL.k3*GLOABAL.dt*Ves
        Vs = Vs - GLOABAL.k1*GLOABAL.dt*Ve*Vs + GLOABAL.k2*GLOABAL.dt*Ves
        Ves = Ves + GLOABAL.k1*GLOABAL.dt*Ve*Vs - GLOABAL.k3*GLOABAL.dt*Ves - GLOABAL.k2*GLOABAL.dt*Ves
        Vp = Vp + GLOABAL.k3*GLOABAL.dt*Ves

        if Vs<=GLOABAL.eps:
            break

    # Draw the curve
    plt.plot(Ve_list)
    plt.plot(Vs_list)
    plt.plot(Ves_list)
    plt.plot(Vp_list)
    plt.legend(["Value_E","Value_S","Value_ES","Value_P"])
    plt.title('Results Figure')
    plt.savefig('./results.png')