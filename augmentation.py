#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from transformations import random_noise,pitch_shift,timestretch
from pathlib import Path
import librosa
import numpy as np


# In[21]:


list_functions=[timestretch,pitch_shift,random_noise]#list of functions


# In[24]:


def augmentation(audio_wav,overlap=0.3):
    '''
    splitting 30 second clip into multiple 10 second clips,
    augmenting clips using timestretching,pitchshifting and
    adding gaussian noise 
    '''
    
    audio_shape=audio_wav.shape[0]
    size=int(audio_shape*(0.3333))#approx. 10 second clip
    step=int(size*(1-overlap))
    short_clips = [audio_wav[i : i + size] for i in range(0, audio_shape+step-size, step)]#overlap starts from step

    count=0
    clips_desired=np.random.choice([1,2,3,4],1,p=[0.4,0.3,0.2,0.1])[0]
    while count<clips_desired:
        short_clip=random.choice(short_clips)
        count+=1
        aug_signal=random.choice(list_functions)(short_clip)#invoke function
        short_clips.append(aug_signal)

        
    return short_clips    


# In[ ]:


if __name__=="__main__" :
    path=Path("genres")
    x=path/"jazz"/"jazz.00008.au"
    wave,sr=librosa.load(x)
    a=augmentation(wave)   


# In[ ]:




