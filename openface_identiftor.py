import tensorflow as tf
from scipy import misc
from facenet_packages import  detect_face,facenet
import subprocess

import time,dlib

start = time.time()

import argparse
import cv2
import os
import pickle
import sys

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
import warnings

import pickle
from operator import itemgetter
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore")


class openface_identiftor:
    def __init__(self):
        fileDir = os.path.dirname(os.path.realpath(__file__))
        modelDir = os.path.join(fileDir, 'openface_models')
        dlibModelDir = os.path.join(modelDir, 'dlib')
        openfaceModelDir = os.path.join(modelDir, 'openface')
        #dlib and openface initial
        self.classifierModel = 'openface_models/generated-embeddings/classifier.pkl'
        self.dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
        self.networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7')
        self.imgDim = 96
        self.threshold = 0.5
        self.cuda = False
        self.verbose = False
        self.initclassifier()

    def train(self, workdir = './openface_models/generated-embeddings'):
        print("openface SVM Classinfer Training Start")
        print("Loading embeddings.")
        fname = "{}/labels.csv".format(workdir)
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
        labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
        fname = "{}/reps.csv".format(workdir)
        embeddings = pd.read_csv(fname, header=None).as_matrix()

        labels=list(labels)
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))

        #use LinearSvm
        clf = SVC(C=1, kernel='linear', probability=True)
        clf.fit(embeddings, labelsNum)
        fName = "{}/classifier.pkl".format(workdir)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'wb') as f:#with open(fName, 'w')
            pickle.dump((le, clf), f)

    def Load_Model(self):
        self.align = openface.AlignDlib(self.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.networkModel,imgDim=self.imgDim,cuda=self.cuda)
      

    def initclassifier(self):
         with open(self.classifierModel, 'rb') as f:  # Use binary processing (rb)
            if sys.version_info[0] < 3:
                (self.le, self.clf) = pickle.load(f)  # le - label and clf - classifer
            else:
                (self.le, self.clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    def classifier_train(self, inputdir='openface_dataset/pre_img'):
        print("openface SVM Classinfer Training Start")
        subprocess.call('../openface/batch-represent/main.lua -outDir ./openface_models/generated-embeddings/ -data %s'%(inputdir),shell = True)
        
        if os.path.isfile('./openface_dataset/pre_img/cache.t7'):
           os.remove('./openface_dataset/pre_img/cache.t7')
        self.train()

    def infer(self,frame,bounding_box):
        identification_result = dict()
        rep, bb = self.getRep(frame,bounding_box)


        predictions = self.clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)

        person = self.le.inverse_transform(maxI)
        confidence = predictions[maxI]

        if confidence > 0.0:
            identification_result['probabilities'] = round(float(confidence), 2)
            identification_result['id'] = person
            identification_result['probabilitie_list'] = list(predictions)

        if not identification_result:
            identification_result['probabilities'] = 0.0
            identification_result['id'] = '_unknown'

        return identification_result



    def getRep(self,frame,bounding_box):

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        #frame = frame[:, :, 0:3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = bounding_box[0]
        bb[1] = bounding_box[1]
        bb[2] = bounding_box[2]
        bb[3] = bounding_box[3]

        rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        alignface = self.align.align(self.imgDim,frame, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        rep = self.net.forward(alignface)

        return (rep, rect)

if __name__=='__main__':
        openface_identiftor().classifier_train()



