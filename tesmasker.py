import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time,sys
import amg8833_i2c
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
model = load_model("maskerfix1.h5")


t0 = time.time()
sensor = []
while (time.time()-t0)<1: # wait 1sec for sensor to start
    try:
        # AD0 = GND, addr = 0x68 | AD0 = 5V, addr = 0x69
        sensor = amg8833_i2c.AMG8833(addr=0x69) # start AMG8833
    except:
        sensor = amg8833_i2c.AMG8833(addr=0x68)
    finally:
        pass
time.sleep(0.1) # wait for sensor to settle
if sensor==[]:
    print("No AMG8833 Found - Check Your Wiring")
    sys.exit(); # exit the app if AMG88xx is not found 


pix_res = (8,8) # pixel resolution
xx,yy = (np.linspace(0,pix_res[0],pix_res[0]),
                    np.linspace(0,pix_res[1],pix_res[1]))
zz = np.zeros(pix_res) # set array with zeros first
# new resolution
pix_mult = (20,15) # multiplier for interpolation 
interp_res = (int(pix_mult[0]*pix_res[0]),int(pix_mult[1]*pix_res[1]))
grid_x,grid_y = (np.linspace(0,pix_res[0],interp_res[0]),
                            np.linspace(0,pix_res[1],interp_res[1]))
# interp function
def interp(z_var):
    # cubic interpolation on the image
    # at a resolution of (pix_mult*8 x pix_mult*8)
    f = interpolate.interp2d(xx,yy,z_var,kind='cubic')
    return f(grid_x,grid_y)
grid_z = interp(zz) # interpolated image


# face_clsfr=cv2.CascadeClassifier('H:/projek masker/haarcascade_frontalface_default.xml')
labels_dict={1:'without_mask',0:'with_mask'}
color_dict={1:(0,0,255),0:(255,0,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

pix_to_read = 64 # read all 64 pixels


while True:
    try:
        (rval, im) = webcam.read()
        # if rval:
        #     continue
        # else:
        #     sys.exit()
        status,pixels = sensor.read_temp(pix_to_read) # read pixels with status
        
        if status: # if error in pixel, re-enter loop and try again
            continue
        
        T_thermistor = sensor.read_thermistor() # read thermistor temp

        # fig.canvas.restore_region(ax_bgnd) # restore background (speeds up run)
        new_z = np.fliplr(interp(np.reshape(pixels,pix_res)))
        # print(new_z)
        im=cv2.flip(im,1,1) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            (x1,y1,w1,h1) = f
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            # face_img = im[x:x+w,y:y+h]
            temperature = new_z[x1:x1+w1, y1:y1+h1]
            # print(temperature)

            temperature = np.round(np.max(temperature.flatten()),1)
            resized=cv2.resize(face_img,(224,224))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,224,224,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
            #print(result)
            
            label=np.argmax(result,axis=1)[0]
            # print(label)
        
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, "{},{}C".format(labels_dict[label],temperature), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            
        # Show the image
        cv2.imshow('masker',   im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key
            break
    except Exception as e:
        print(e)
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()