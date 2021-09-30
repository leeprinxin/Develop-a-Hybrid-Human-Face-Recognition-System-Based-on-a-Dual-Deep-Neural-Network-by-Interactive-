from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle, dlib, openface
import time, random, csv

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from facenet_packages import facenet, detect_face
from facenet_identiftor import facenet_identiftor
from openface_identiftor import openface_identiftor

from sklearn.metrics import confusion_matrix

class facedetector:
    def __init__(self):
        self.npy = ''
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, self.npy)
                self.minsize = 15  # minimum size of face
                self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                self.factor = 0.709  # scale factor
                self.margin = 44
                self.frame_interval = 3
                self.batch_size = 1000
                self.image_size = 182, 96
                self.input_image_size = 160

                #需要使用dlib進行才切與對齊  才支援openface訓練格式
                fileDir = os.path.dirname(os.path.realpath(__file__))
                modelDir = os.path.join(fileDir, 'openface_models')
                dlibModelDir = os.path.join(modelDir, 'dlib')
                openfaceModelDir = os.path.join(modelDir, 'openface')
                self.dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
                self.networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7')
                self.imgDim = 96
                self.cuda = False
                self.align = openface.AlignDlib(self.dlibFacePredictor)
                self.net = openface.TorchNeuralNet(self.networkModel,imgDim=self.imgDim,cuda=self.cuda)

        
    def getfacebounding_box(self,frame):
        frame = cv2.resize(frame, (640, 360))  # resize frame (optional)
        dets = []
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Detected_FaceNum: %d' % nrof_faces)
        
     

        if nrof_faces > 0:
            dets = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame.shape)[0:2]
            if nrof_faces > 1:
               bounding_box_size = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
               img_center = img_size / 2
               offsets = np.vstack([(dets[:, 0] + dets[:, 2]) / 2 - img_center[1], (dets[:, 1] + dets[:, 3]) / 2 - img_center[0]])
               offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
               index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
               dets = [dets[index, :]]
           
            for det in dets:
               if det[0] < 0 or det[1] < 0 or det[2] < 0 or det[3] < 0:
                  print('no....')
                  return [],frame    
        return dets,frame

    def get_crop_faceimages(self, bounding_box, frame):
        frame = cv2.resize(frame, (640, 360))  # resize frame (optional)
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]

        cropped = []
        scaled = []
        scaled_reshape = []
        bb = np.zeros(4, dtype=np.int32)

        bb[0] = bounding_box[0]
        bb[1] = bounding_box[1]
        bb[2] = bounding_box[2]
        bb[3] = bounding_box[3]

        

        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
        cropped = facenet.flip(cropped, False)
        scaled = misc.imresize(cropped, (self.image_size[0], self.image_size[0]), interp='bilinear')
         
        
        rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        alignface = self.align.align(self.image_size[1],frame, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        scaled2 = cv2.cvtColor(alignface, cv2.COLOR_RGB2BGR)

        return [scaled, scaled2]

def loging(results, Accuracys, Victorys, index_dict, y_true, y_pred):

    with open('p_met.txt','w', newline='') as csvfile:
        p_met = []
        for result in results:
            p_met.append(result['Actual_Result_list'])
        csvfile.write(str(p_met))

    with open('L_met.txt','w', newline='') as csvfile:
        l_met = []
        for result in results:
            l_met.append(result['probabilitie_list'])
        csvfile.write(str(l_met))
            
    with open('log.csv','w', newline='') as csvfile:
        fieldnames = ['id', 'facenet_id_res', 'facenet_prob_res', 'openface_id_res', 'openface_prob_res', 'Pred_Result_id','Pred_Result_prob','Actual Result','Update Result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            facenet_id_res, facenet_prob_res,openface_id_res, openface_prob_res, Pred_Result_id, Pred_Result_prob, Actual_Result , Update_Result= result['facenet_id_res'],result['facenet_prob_res'],result['openface_id_res'], result['openface_prob_res'], result['Pred_Result_id'], result['Pred_Result_prob'], result['Actual_Result'],result['Update_Result']
            writer.writerow({fieldnames[0]:result['id'],
                             fieldnames[1]:facenet_id_res,
                             fieldnames[2]:facenet_prob_res,
                             fieldnames[3]:openface_id_res,
                             fieldnames[4]:openface_prob_res,
                             fieldnames[5]:Pred_Result_id,
                             fieldnames[6]:Pred_Result_prob,
                             fieldnames[7]:Actual_Result,
                             fieldnames[8]:Update_Result})

    with open('log.txt','w') as csvfile:
        csvfile.write(str(index_dict)+'\r\n')
        for k in Accuracys:
            csvfile.write('k=%s accuracy=%s \r\n'%(k,Accuracys[k][0]))
            csvfile.write('Confusion : \r\n%s' % (Accuracys[k][1]))
            csvfile.write('\r\n')

    with open('victorys.txt','w') as csvfile:
        csvfile.write('Victory: ' + str(Victorys) + '\r\n')
        csvfile.write('y_true: ' + str(y_true) + '\r\n')
        csvfile.write('y_pred: ' + str(y_pred) + '\r\n')


def get_all_image_names(rootdir):
    abs_image_names = []
    image_names = []
    folders = os.listdir(rootdir)
    print(folders)
    folders.sort()

    for folder in folders:
      #if not folder == 'b11': continue 
      names = os.listdir(os.path.join(rootdir,folder))
      abs_image_names = abs_image_names+[os.path.join(rootdir,folder,name) for name in names ]
    
    random.seed(100)
    random.shuffle(abs_image_names)
    image_names = [os.path.basename(abs_image_name) for abs_image_name in abs_image_names]
    return abs_image_names, image_names
  
def get_index_dict(dir):
    my_index_dict  = dict()
    names = os.listdir(dir)
    names.sort()
    for index, myid in enumerate(names):
        my_index_dict[myid] = index
    return my_index_dict
    

def getModel():
    myfacedetector = facedetector()
    f_identiftor = facenet_identiftor()
    f_identiftor.Load_Model()
    o_identiftor = openface_identiftor()
    o_identiftor.Load_Model()
    return myfacedetector, f_identiftor, o_identiftor

def run(dir):
    myfacedetector, f_identiftor, o_identiftor = getModel()
    abs_image_names, names  = get_all_image_names(dir)
    index_dict = get_index_dict('facenet_dataset/train_img')
    print(index_dict)
    y_true = [index_dict[name[0:name.index('_')]] for name in names]
    y_pred = []
    victory_ratio = []
    results = []#log csv detial
    Accuracys = dict()#log txt accuracy and confision
    count = 0
    train_c = 0
    threshold = 0.3
    for i,image_name in enumerate(abs_image_names):
        
        print(i,'----',image_name)
        image = cv2.imread(image_name, 0)#做判斷用的
        mtcnn_image = misc.imread(image_name)#儲存用

        # step 1. gettar image face position
        dets, image = myfacedetector.getfacebounding_box(image)
        if dets == [] :
            y_true[i] = 'x'
            y_pred.append('x')
        print(dets)
        for det in dets:
            count += 1 #辨識影像次數
            result = dict() #紀錄識別情況明細CSV
            # step 2. gettar image face recognition result
            f_identification_result = f_identiftor.infer(image, det)
            o_identification_result = o_identiftor.infer(image, det)
            croped_list = myfacedetector.get_crop_faceimages(det,mtcnn_image)
            
           
            print('bb: %s' % (det))
            print('facenet model result: %s' % (f_identification_result))
            print('openface model result: %s' % (o_identification_result))

            # step 5. log to csv
            result['id'] = count
            result['facenet_id_res'] = str(f_identification_result['id'])
            result['openface_id_res'] = str(o_identification_result['id'])
            result['facenet_prob_res'] = str(f_identification_result['probabilities'])
            result['openface_prob_res'] = str(o_identification_result['probabilities'])
            name = names[i]
            result['Actual_Result'] = str(name[0:name.index('_')])
            Actual_Result_list = [0]*len(index_dict)
            Actual_Result_list[index_dict[result['Actual_Result']]] = 1
            result['Actual_Result_list'] = Actual_Result_list
            

            if train_c > 1200:
                 throshold = 0.7 #0.7 0.63 0.5 0.4 0.3, 0.7 0.63 0.55 0.5 0.3 0.25
            elif train_c > 1000:
                 throshold = 0.63
            elif train_c > 800:
                 throshold = 0.55
            elif train_c > 500:
                 throshold = 0.45
            elif train_c > 220:
                 throshold = 0.25
            else :
                 throshold = 0.15

            # step 3. comparay results, and train
            result['Update_Result'] = comparay2(f_identification_result, o_identification_result, croped_list, throshold)
            if not result['Update_Result'] == 'o-f<0.63' or not result['Update_Result'] == 'c not= o-f>0.63':
                train_c += 1

            
            # step 4. select best model
            if f_identification_result['probabilities'] > o_identification_result['probabilities']:
                victory_ratio.append('f')
                y_pred.append(index_dict[f_identification_result['id']])
                result['Pred_Result_id'] = f_identification_result['id']
                result['Pred_Result_prob'] = f_identification_result['probabilities']
                result['probabilitie_list'] = f_identification_result['probabilitie_list']
            elif f_identification_result['probabilities'] < o_identification_result['probabilities']:
                victory_ratio.append('p')
                y_pred.append(index_dict[f_identification_result['id']])
                result['Pred_Result_id'] = f_identification_result['id']
                result['Pred_Result_prob'] = f_identification_result['probabilities']
                result['probabilitie_list'] = f_identification_result['probabilitie_list']
            else:
                victory_ratio.append('d')
                y_pred.append(index_dict[f_identification_result['id']])
                result['Pred_Result_id'] = f_identification_result['id']
                result['Pred_Result_prob'] = f_identification_result['probabilities']
                result['probabilitie_list'] = f_identification_result['probabilitie_list']

            results.append(result)

            #time.sleep(1)
            # step 6. train
            print('starting train model....')
            if train_c%30 == 0:
                train_model(f_identiftor, o_identiftor, fpredir = './facenet_dataset/pre_img', opredir='./openface_dataset/pre_img')
               

            if count%100 == 0:
                tmpy_true = list(filter(lambda x: not x == 'x', y_true))
                tmpy_pred = list(filter(lambda x: not x == 'x', y_pred))
                Accuracys[count] = [np.mean(np.equal(tmpy_true[:count], tmpy_pred)), confusion_matrix(tmpy_true[:count], tmpy_pred, labels=list(index_dict.values()))]
                victory = 'f wins: '+ str(victory_ratio.count('f'))+', o wins: '+str(victory_ratio.count('p'))+ ', d : '+str(victory_ratio.count('d'))
                loging(results,Accuracys,victory,index_dict,tmpy_true[:count],tmpy_pred)#防止中斷未寫入

    y_true = list(filter(lambda x: not x == 'x', y_true))
    y_pred = list(filter(lambda x: not x == 'x', y_pred))
    Accuracys[count] = [np.mean(np.equal(y_true, y_pred)), confusion_matrix(y_true, y_pred, labels=list(index_dict.values()))]
    print('混淆矩陣:\n',confusion_matrix(y_true, y_pred, labels=list(index_dict.values())))
    print('準確率：',np.mean(np.equal(y_true, y_pred)))
    victory = 'f wins: '+ str(victory_ratio.count('f'))+', o wins: '+str(victory_ratio.count('p'))+ ', d : '+str(victory_ratio.count('d'))
    print(victory)
    loging(results,Accuracys,victory,index_dict,y_true,y_pred)
    

def train_model(f_identiftor, o_identiftor, fpredir=None, opredir=None):
    if fpredir:
       f_identiftor.classifier_train(fpredir)
       f_identiftor.initclassifier()
    '''if opredir:
       o_identiftor.classifier_train(opredir)
       o_identiftor.initclassifier()'''

def add_train_image(fileName, rep, label):
    
    train_dir = ['facenet_dataset/pre_img', 'openface_dataset/pre_img']
    label_train_dir = [os.path.join(train_dir[0], label), os.path.join(train_dir[1], label) ]
    if not os.path.isdir(label_train_dir[0]):
        os.mkdir(label_train_dir[0])
    if not os.path.isdir(label_train_dir[1]):
        os.mkdir(label_train_dir[1])
    
    misc.imsave(os.path.join(label_train_dir[0], fileName), rep[0])
    cv2.imwrite(os.path.join(label_train_dir[1], fileName), rep[1])
    

def add_observed_image(fileName, rep):
    train_dir = os.path.join('observed_images',fileName)
   
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    misc.imsave(os.path.join(train_dir, fileName), rep[0])
    cv2.imwrite(os.path.join(train_dir, fileName+'_2'), rep[1])

def comparay2(f_identification_result, o_identification_result,crop, threshold=0.3):
    f_id = f_identification_result['id']
    if f_identification_result['probabilities'] >=threshold:
       add_train_image('%s_%d.png' % (f_id, time.time()), crop,f_id)#都訓練
       return 'c= o-f>0.63'
    else:
        return 'o-f<0.63'

def comparay(f_identification_result, o_identification_result, crop, threshold=0.3,actual_result=None):
    
    f_id, o_id = actual_result, actual_result#f_identification_result['id'], o_identification_result['id']
   
    if f_identification_result['probabilities'] >=threshold and o_identification_result['probabilities'] >=threshold:

       if f_identification_result['id'] == o_identification_result['id'] :
           add_train_image('%s_%d.png' % (f_id, time.time()), crop,f_id)#都訓練
           return 'c= o-f>0.63'
       elif not f_identification_result['id'] == o_identification_result['id'] :
           add_observed_image('%s_%s_%d.png' % (f_id, o_id, time.time()), crop)#待觀察
           return 'c not= o-f>0.63'
    elif f_identification_result['probabilities'] < threshold and o_identification_result['probabilities'] < threshold:
        print('throw away.')#拋棄影像
        return 'o-f<0.63'
    elif f_identification_result['probabilities'] >= threshold and o_identification_result['probabilities'] < threshold:
        add_train_image('%s_%d.png' % (f_id, time.time()), crop, f_id)#依照facenet id
        return 'c = f>o'
    elif f_identification_result['probabilities'] < threshold and o_identification_result['probabilities'] >= threshold:
        add_train_image('%s_%d.png' % (o_id, time.time()), crop, o_id)#依照openface id
        return 'c = f<o'


if __name__=='__main__':
    start = time.time()
    run('/media/pingxin/ADATA UFD/訓練測試集')
    end = time.time()
    print('time elapsed: ' + str(round(end-start, 2)) + ' seconds')


