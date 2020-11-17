import numpy as np
import sys,os
import cv2
caffe_root = '/home/pczee211008/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import time




FILE_OUTPUT = '/home/pczee211008/mobilenet_ssd_pedestrian_detection/output1.mp4'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file
cap = cv2.VideoCapture('/home/pczee211008/mobilenet_ssd_pedestrian_detection/Pédestrians Vs Traffic on the road # 4.mp4')

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
output = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
10, (frame_width, frame_height))


net_file= 'deploy.prototxt'
caffe_model='mobilenet_iter_73000.caffemodel'

#video_capture = cv2.VideoCapture(0)

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)

CLASSES = ('background',
           'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
           'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase'
,'frisbee'
,'skis'
,'snowboard'
,'sports ball'
,'kite'
,'baseball bat'
,'baseball glove'
,'skateboard'
,'surfboard'
,'tennis racket'
,'bottle'
,'wine glass'
,'cup'
,'fork'
,'knife'
,'spoon'
,'bowl'
,'banana'
,'apple'
,'sandwich'
,'orange'
,'broccoli'
,'carrot'
,'hot dog'
,'pizza'
,'donut'
,'cake'
,'chair'
,'couch'
,'potted plant'
,'bed'
,'dining table'
,'toilet'
,'tv'
,'laptop'
,'mouse'
,'remote'
,'keyboard'
,'cell phone'
,'microwave'
,'oven'
,'toaster'
,'sink'
,'refrigerator'
,'book'
,'clock'
,'vase'
,'scissors'
,'teddy bear'
,'hair drier'
,'toothbrush')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect():
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, origimg = cap.read()
    #ret, origimg = video_capture.read()
        img = preprocess(origimg)

        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        net.blobs['data'].data[...] = img
        start = time.time()
        out = net.forward()
        use_time=time.time() - start
        print("time="+str(use_time)+"s")
        box, conf, cls = postprocess(origimg, out)

        for i in range(len(box)):
            if conf[i] > 0.3:
                p1 = (box[i][0], box[i][1])
                p2 = (box[i][2], box[i][3])
                cv2.rectangle(origimg, p1, p2, (0,255,0))
                p3 = (max(p1[0], 15), max(p1[1], 15))
                title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
                cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        if ret == True:
            #Saves for video
            output.write(origimg)

        cv2.imshow("SSD", origimg)

        cv2.waitKey(1) & 0xff
            #Exit if ESC pressed
        return True
        cap.release()
        output.release()

if __name__ == '__main__':


    while True:
        detect()

