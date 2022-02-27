from easydict import EasyDict

GLOABAL = dict(
    # initialize concentration
    Ve=1,
    Vs=10, 
    Ves=0, 
    Vp=0,
    # initialize rate constant
    k1=100,
    k2=600, 
    k3=150,
    # Set the unit time
    dt=1e-5,
    # set the minimum value
    eps=1e-6
)
GLOABAL = EasyDict(GLOABAL)
MAX_EPOCHS = 100000