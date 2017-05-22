import environment as p
import gym
import scipy.misc as sc
import numpy as np

env = gym.make('Breakout-v0')

preproc = p.Preprocessing(env)

env.step(2)
env.step(3)
env.step(1)
obs, rw, done, inf = env.step(2)

img_gray = preproc.map2Y(obs)

# experiments with cropping
img_cropped = np.zeros((185, 160, 1))
img_cropped[:,:,0] = img_gray[16:201,:,0]

img = np.zeros((84, 84))
img[:, :] = preproc.rescale(img_cropped)[:,:,0]

sc.imsave('test.png',img)