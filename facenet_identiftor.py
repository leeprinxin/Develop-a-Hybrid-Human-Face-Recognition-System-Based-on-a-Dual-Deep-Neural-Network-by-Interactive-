from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from facenet_packages import facenet
from facenet_packages.classifier import training


class facenet_identiftor:
  def __init__(self):
      self.input_video = 0  # "政見會互嗆總機.讀稿機.mp4"
      self.modeldir = './facenet_models/20180402-114759/20180402-114759.pb'
      self.classifier_filename = './facenet_models/class/classifier.pkl'
      self.npy = ''
      self.train_img = './facenet_dataset/train_img'

  def Load_Model(self):
     with tf.Graph().as_default():
         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
         self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
         with self.sess.as_default():

             self.frame_interval = 3
             self.image_size = 182
             self.input_image_size = 160
             self.HumanNames = os.listdir(self.train_img)
             self.HumanNames.sort()
             print('Loading Modal')
             facenet.load_model(self.modeldir)
             self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
             self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
             self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
             self.embedding_size = self.embeddings.get_shape()[1]
             self.initclassifier()

            
  def initclassifier(self):
       classifier_filename_exp = os.path.expanduser(self.classifier_filename)
       with open(classifier_filename_exp, 'rb') as infile:
        (self.model, self.class_names) = pickle.load(infile)

  def classifier_train(self, inputdir):
      print("facenet SVM Classinfer Training Start")
      obj = training(inputdir, self.modeldir, self.classifier_filename )
      get_file = obj.main_train()
      print('Saved facenet classifier model to file "%s"' % get_file)


  def infer(self,frame,bounding_box):

      identification_result = dict()

      if frame.ndim == 2:
          frame = facenet.to_rgb(frame)
      frame = frame[:, :, 0:3]

      cropped = []
      scaled = []
      scaled_reshape = []
      bb = np.zeros( 4, dtype=np.int32)

      try:
          emb_array = np.zeros((1, self.embedding_size))
          bb[0] = bounding_box[0]
          bb[1] = bounding_box[1]
          bb[2] = bounding_box[2]
          bb[3] = bounding_box[3]


          cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
          cropped = facenet.flip(cropped, False)

          scaled_temp = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear') # 暫存
          scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
          scaled = cv2.resize(scaled, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
          scaled = facenet.prewhiten(scaled)

          scaled_reshape = scaled.reshape(-1, self.input_image_size, self.input_image_size, 3)
          feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
          emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
          predictions = self.model.predict_proba(emb_array)
          # print(predictions) 顯示所有訓練類別對於此張影像信心度
          best_class_indices = np.argmax(predictions, axis=1)
          best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

          # print(best_class_indices, ' with accuracy ', best_class_probabilities) 顯示信心度最高的類別index
          # print(best_class_probabilities) 與信心度
          if best_class_probabilities > 0.0:
                for H_i in self.HumanNames:
                      if self.HumanNames[best_class_indices[0]] == H_i:
                          identification_result['probabilities'] = round(float(best_class_probabilities[0]),2)
                          identification_result['id'] = self.HumanNames[best_class_indices[0]]
                          identification_result['probabilitie_list'] = predictions.tolist()[0]
                          

      except IndexError:
                print("IndexError: list index out of range")

      if not identification_result:
           identification_result['probabilities'] = 0.0
           identification_result['id'] = '_unknown'

      return identification_result
        
def main():
   facenet_identiftor().classifier_train('./facenet_dataset/pre_img')

if __name__ == '__main__':
   main()

    
    

