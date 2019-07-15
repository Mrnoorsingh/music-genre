#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import librosa


# In[2]:


def timestretch(y):
    rate=random.uniform(0.1,4)
    y_new=librosa.effects.time_stretch(y,rate=rate)
    return y_new

def pitch_shift(y):
    shift=random.uniform(-5,5)
    y_new=librosa.effects.pitch_shift(y,sr=22050,n_steps=shift)
    return y_new
    
    
def random_noise(y):
    noise = np.random.normal(0,1,y.shape[0])
    noise=noise/100
    noisy_audio=y+noise
    return noisy_audio

