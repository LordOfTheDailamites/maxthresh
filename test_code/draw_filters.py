import numpy as np
import cv2

FILTER_ADDRESS = "..\\filters\\weighted_softmax_chak_bgr_8_filter.npy"
C = 3

def bayer_filter(filter_size):
    bayer = np.array([[[1,0,0],[0,1,0]],
                      [[0,1,0],[0,0,1]]])
    size = round(filter_size/2)
    bayer = np.tile(bayer, (size,size,1))
    return bayer

def lukac_filter(filter_size):
    lukac = np.array([[[0,1,0],[0,0,1]],
                      [[0,1,0],[1,0,0]],
                      [[0,0,1],[0,1,0]],
                      [[1,0,0],[0,1,0]]])
    size = round(filter_size/2)
    lukac = np.tile(lukac, (int(size/2),size,1))
    return lukac

def rgbw_filter(filter_size):
    rgbw = np.array([[[1,0,0,0],[0,1,0,0]],
                      [[0,0,0,1],[0,0,1,0]]])
    size = round(filter_size/2)
    rgbw = np.tile(rgbw, (size,size,1))
    return rgbw

def cfz_filter(filter_size):
    cfz = np.asarray([[[0,1,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]],
                      [[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0]],
                      [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
                      [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]])
    size = round(filter_size/4)
    cfz = np.tile(cfz, (size,size,1))
    return cfz

def load_filter(filter_address):
    learned_filter = np.load(filter_address)[0]
    return learned_filter

binary_filter = load_filter(FILTER_ADDRESS)
num_pixel = binary_filter.shape[0] * binary_filter.shape[1]
num_red = len(np.where(binary_filter[:,:,-1]==1)[0])
print("Ratio of red pixels: {}".format(100 * num_red/num_pixel))
num_green = len(np.where(binary_filter[:,:,-2]==1)[0])
print("Ratio of green pixels: {}".format(100 * num_green/num_pixel))
num_blue = len(np.where(binary_filter[:,:,-3]==1)[0])
print("Ratio of blue pixels: {}".format(100 * num_blue/num_pixel))
if C == 4: 
    num_white = len(np.where(binary_filter[:,:,0]==1)[0])
    print("Ratio of white pixels: {}".format(100 * num_white/num_pixel))
    
if C == 3:
    binary_filter_b = np.expand_dims(binary_filter[:,:,0], axis=2)
    binary_filter_g = np.expand_dims(binary_filter[:,:,1], axis=2)
    binary_filter_r = np.expand_dims(binary_filter[:,:,2], axis=2)
    
    binary_filter_visualized = 255*np.concatenate((binary_filter_b,binary_filter_g,binary_filter_r), axis=2)
elif C == 4:
    binary_filter_c = np.tile(np.expand_dims(binary_filter[:,:,0], axis=2),(1,1,3))
    binary_filter_b = np.expand_dims(binary_filter[:,:,1], axis=2)
    binary_filter_g = np.expand_dims(binary_filter[:,:,2], axis=2)
    binary_filter_r = np.expand_dims(binary_filter[:,:,3], axis=2)
    
    binary_filter_visualized = 255*np.logical_or(binary_filter_c, 
                                                 np.concatenate((binary_filter_b,binary_filter_g,binary_filter_r), axis=2))

binary_filter_visualized = binary_filter_visualized[2:6,2:6,:]
cv2_learned_filter = cv2.resize(binary_filter_visualized.astype("uint8"), (560,560), interpolation=cv2.INTER_AREA).astype("uint8")
cv2_learned_filter = np.concatenate([np.pad(cv2_learned_filter[:,:,0], ((2, 2),(2, 2)), 'constant')[:,:,np.newaxis],
     np.pad(cv2_learned_filter[:,:,1], ((2, 2),(2, 2)), 'constant')[:,:,np.newaxis],
     np.pad(cv2_learned_filter[:,:,2], ((2, 2),(2, 2)), 'constant')[:,:,np.newaxis]], axis=2)
cv2.imshow("Learned Filter", cv2_learned_filter)
cv2.waitKey(0)

cv2.imwrite("filter_linear_henz_bgr_8.png", cv2_learned_filter)