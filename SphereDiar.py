from sklearn.utils import linear_assignment_
import numpy as np
from spherecluster import SphericalKMeans
from sklearn.metrics import silhouette_score
import os
import glob
from joblib import Parallel, delayed
import multiprocessing
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from keras.models import *
from keras.layers import *
from librosa.feature import *
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from librosa.util import frame
import warnings
import wavefile

from utils import feature_extractor , reorganize_lab , silh_score , DER , Top2S


# SPHEREDIAR: SPEAKER DIARIZATION SYSTEM

class SphereDiar():
    
    def __init__(self,pathToAudioFile,pathToModel):
        self.embeddings_ = []  
        self.speaker_labels_ = []
        self.emb_2d_ = []
        self.X_ = []
        self.centers_ = {}
        self.opt_speaker_num_ = 0

        (self.rate,self.sig) = wavefile.load(pathToAudioFile)
        self.sig = np.squeeze(self.sig)
        dir = os.path.dirname(__file__)
        # SS_model = load_model( os.path.join(dir,"Model/SphereSpeaker.hdf"))
        SS_model = load_model(pathToModel)
        # Exclude softmax layer
        SS = Model(inputs=SS_model.input,
                            outputs=SS_model.layers[-2].output)
        self.SS_ = SS
               
    def extract_features(self, signal, frame_len = 2, hop_len = 0.5, fs = 16000):  
        
        # Frame duration 2s, overlap duration 1.5s, assuming 16 kHz sampling rate
        S = np.transpose(frame(signal, int(frame_len*fs), int(hop_len*fs)))
        
        # 201 sequences of 59 dimensional MFCC based features
        X = list(map(lambda s: feature_extractor(s, fs), S))
        X = np.swapaxes(X, 1, 2)
        self.X_ = X
        return X
        
    def get_embeddings(self, X = []):
               
        if (len(self.X_) == 0) and (len(X) == 0):
            raise RuntimeError("No features available.")

        elif len(X) != 0:
             self.X_ = X
            
        embeddings = self.SS_.predict(self.X_)       
        self.embeddings_ = embeddings
        return embeddings
        
        
    def cluster(self, rounds = 20, clust_range = [2,12], num_cores = 1, threshold = 0.1, embeddings = []):
        
        
        if (len(self.embeddings_) == 0) and (len(embeddings) == 0):
            raise RuntimeError("No speaker embeddings available.")
            
        # If embeddings are not given
        if len(embeddings) == 0:
            embeddings = self.embeddings_
            
        else:
            self.embeddings_ = embeddings
            
                     
        # Top Two Silhouettes
        opt_center_num, center_dict = Top2S(embeddings, clust_range = clust_range, 
                                       rounds = rounds, num_cores = num_cores, threshold = threshold)
        self.centers_ = center_dict
        self.opt_speaker_num_ = opt_center_num   
        
        # Get speaker labels 
        spkmeans = SphericalKMeans(n_clusters=len(center_dict[opt_center_num]), 
                                                   init = center_dict[opt_center_num], 
                                                   max_iter=1, n_init=1, n_jobs=1).fit(embeddings)  
        self.speaker_labels_ = spkmeans.labels_+1 
    
    def visualize(self, indices = [], center_num = 0, 
                  ref_labels = [], use_colors = True):
        
        
        # If indices are not given
        if len(indices) ==0:
            indices = np.arange(len(self.embeddings_))
        
        # If center number is not given
        if center_num == 0:
            center_num = self.opt_speaker_num_
                
        # If reference labels are used
        if len(ref_labels) != 0:
            speaker_labels = ref_labels   
            
        # Allow visualization of different center number configurations
        else:        
            # Get speaker labels 
            spkmeans = SphericalKMeans(n_clusters=len(self.centers_[center_num]), 
                                                       init = self.centers_[center_num], 
                                                       max_iter=1, n_init=1, n_jobs=1).fit(self.embeddings_[indices])  
            speaker_labels = spkmeans.labels_+1 
        
        
        if len(self.speaker_labels_) == 0:
            raise RuntimeError("Clustering not performed.")
                                       
        # Compute TSNE only once
        if len(self.emb_2d_) == 0:
            
            print("Computing TSNE transform...")
            tsne = TSNE(n_jobs=4)
            self.emb_2d_ = tsne.fit_transform(self.embeddings_)
        
        
        # Visualize
        emb_2d = self.emb_2d_[indices]
        speaker_labels = speaker_labels.astype(np.int)
        speakers = np.unique(speaker_labels)
        colors=cm.rainbow(np.linspace(0,1,len(speakers)))
        plt.figure(figsize=(7,7))

        for speaker in speakers:

            speak_ind = np.where(speaker_labels == speaker)[0]
            x, y = np.transpose(emb_2d[speak_ind])
            if use_colors == True:
               plt.scatter(x, y, c="k", edgecolors=colors[speaker-1], s=2,  label=speaker)
            else:
               plt.scatter(x, y, c="k", edgecolors="k", s=2,  label=speaker)

        plt.legend(title = "Speakers", prop={'size': 10})

        if len(ref_labels) == 0:
            plt.title("Predicted speaker clusters")
        else:
            plt.title("Reference speaker clusters")  
        plt.show()

    def calc_DER(self, ref_labels, ref_indices):
        
        labels = self.speaker_labels_[ref_indices]        
        der = DER(ref_labels, labels)
        print("DER (%): ", round(der*100, 3))


    def post_Processing(self):

        count = 1 
        minimumLength = 10
        if self.rate >20000:
            minimumLength = 15
        for i in range(1,self.speaker_labels_.shape[0]):
            if self.speaker_labels_[i] == self.speaker_labels_[i-1]:
                count+=1
            else:
                if count < minimumLength and (i-count-1) >=0:
                    for j in range(i-count , i ):
                        self.speaker_labels_[j]= self.speaker_labels_[i-count-1]
                count = 1



        rows , cols = (4,((int)(self.sig.shape[0]/16000*2)))
        diar = np.zeros((rows,cols))
        for i in range(0,self.speaker_labels_.shape[0]):
            for j in range(0,4):
                if((i-j)>=0 and (i-j)<self.speaker_labels_.shape[0]):
                    diar[j,i]= self.speaker_labels_[i-j]




        preFinalDiar = np.zeros(diar.shape[1]-3)
        for i in range(diar.shape[1]-3):
            count = 1
            for j in range(1,4):
                if diar[j,i] == diar[j-1,i] or diar[j,i]==0:
                    count+=1
                else:
                    if count >1  :
                        preFinalDiar[i]=diar[0,i]
                    else:
                        preFinalDiar[i]=diar[3,i]
                    break
                if count == 4:
                    preFinalDiar[i] = diar[0,i]
 


        finalDiar = []
        count = 0.5;
        index = 0
        currentTime = 0
        for i in range(1, preFinalDiar.shape[0]):
            if preFinalDiar[i] == preFinalDiar[i-1]:
                count += 0.5
            else:
                finalDiar.insert(index  ,(preFinalDiar[i-1],currentTime,currentTime+count))
                currentTime = currentTime+count
                index += 1
                count = 0.5
            if i == preFinalDiar.shape[0]-1:
                finalDiar.insert(index , (preFinalDiar[i-1],currentTime,currentTime+count))
                index += 1
                count = 0.5

        finalDiar = np.array(finalDiar)
        for i in range(finalDiar.shape[0]):
            for j in range(1,3):
                finalDiar[i,j] = finalDiar[i,j] *16000/self.rate

        return finalDiar



    def start(self):
        X = self.extract_features(self.sig)
        F = self.get_embeddings()
        self.cluster(embeddings = F, clust_range = [2,8] , rounds = 10)
        #visualize(use_colors = True)
        return self.post_Processing()



  
