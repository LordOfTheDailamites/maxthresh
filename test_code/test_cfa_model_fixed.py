import tensorflow as tf
import numpy as np
import cv2
import os
from math import log10
from skimage.metrics import structural_similarity as find_ssim

# Initializating the addresses
MODEL_ADDRESS = "..\\models_scenario_3\\cfa_demosaicer_bayer_custom_8"
IMAGE_ADDRESS = "..\\dataset\\test_images_bsd500"
FILTER_ADDRESS = "..\\filters\\test_images_kodak"

FILTER_NAME = "Bayer"
DEMOSAICER = "Custom"

# Initializating the constants
NORMALIZE = False
C = 3
P = 8
BATCH_SIZE = 32
IMAGE_SIZE = 3*P

# Calculates PSNR value between two images.
def find_psnr(im1, im2):
    return 20*log10(255/np.sqrt(np.square(im1 - im2).mean()))
# Divides an array of images into smaller blocks.
def divide_to_blocks(image, stride, image_size):
    image_blocks = []
    height, width = image.shape[:2]
    # Calculating the number of blocks per height and width
    block_no_vertical = ((height - image_size) // stride) + 1
    block_no_horizontal = ((width - image_size) // stride) + 1
    # Dividing the image into blocks
    for i in range(block_no_vertical):
        for j in range(block_no_horizontal):
            image_block = image[i*stride:(i*stride) + image_size, j * stride:(j * stride) + image_size]
            image_blocks.append(image_block)
    return image_blocks
# Merges image blocks to create a full image.
def merge_blocks(blocks, height, width):
    construction = np.array([])
    for i in range(height):
        const_width = blocks[width*i]
        for j in range(width-1):
            const_width = np.concatenate((const_width, blocks[width*i+j+1]), 1)
        if i == 0:
            construction = const_width
        else:
            construction = np.concatenate((construction, const_width), 0)
    return construction

if FILTER_NAME == "Bayer":       # 3P x 3P BGGR Bayer Filter
    bayer_filter = np.asarray([[[1,0,0],[0,1,0]],
                               [[0,1,0],[0,0,1]],])
    cfa_filter = np.tile(bayer_filter,(int(IMAGE_SIZE/2),int(IMAGE_SIZE/2),1))
if FILTER_NAME == "Lukac":       # 3P x 3P BGGR Bayer Filter
    lukac_filter = np.array([[[0,1,0],[0,0,1]],
                             [[0,1,0],[1,0,0]],
                             [[0,0,1],[0,1,0]],
                             [[1,0,0],[0,1,0]]])
    cfa_filter = np.tile(lukac_filter,(int(IMAGE_SIZE/2),int(IMAGE_SIZE/2),1))
elif FILTER_NAME == "CFZ":       # 3P x 3P CBGR CFZ Filter
    cfz_filter = np.asarray([[[0,1,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]],
                               [[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0]],
                               [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
                               [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]])
    cfa_filter = np.tile(cfz_filter,(int(IMAGE_SIZE/4),int(IMAGE_SIZE/4),1))
elif FILTER_NAME == "Fixed":    # 3P x 3P Maximum Thresholding Filter
    fixed_filter = np.load(FILTER_ADDRESS)[0]
    cfa_filter = np.tile(fixed_filter,(3,3,1))

# Loading the model
model = tf.keras.models.load_model(MODEL_ADDRESS)
original_images = []
predicted_images = []
image_psnr = []
image_ssim = []
image_num = 0
for root, dirs, files in os.walk(IMAGE_ADDRESS, topdown=True):
    for name in files:
        print("Image name: ", name)
        new_image = cv2.imread(os.path.join(IMAGE_ADDRESS, name))
        if C == 4:
            image_grayscale = np.expand_dims(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY), axis=2)
            new_image = np.concatenate((image_grayscale, new_image), axis=2)
        image_input = np.asarray(divide_to_blocks(new_image,IMAGE_SIZE,IMAGE_SIZE))
        if DEMOSAICER == "Chakrabarti":
            image_output = np.copy(image_input)[:,P:2*P,P:2*P]
        else:
            image_output = np.copy(image_input)
        if C == 4: image_output = image_output[:,:,:,1:]
        height, width = new_image.shape[:2]
        height = ((height - IMAGE_SIZE) // IMAGE_SIZE) + 1
        width = ((width - IMAGE_SIZE) // IMAGE_SIZE) + 1
        original_image = merge_blocks(image_output.astype(int), height, width)
        # Normalizing the dataset
        if NORMALIZE:
            image_input = image_input / 255
            image_output = image_output / 255
        filtered_input = []
        for img in image_input:
            img_filtered = np.multiply(img, cfa_filter)
            img_mask = np.sum(img_filtered, axis=2)
            img_mask = np.expand_dims(img_mask, axis=2)
            filtered_input.append(img_mask)
        image_input_filtered = np.asarray(filtered_input)
        # Converting the input images and output ground truth into TensorFlow dataset object
        image_dataset = tf.data.Dataset.from_tensor_slices((image_input_filtered, image_output)).batch(BATCH_SIZE)

        print("Reconstructing the image from the samples.")
        predictions = model.predict(image_dataset, verbose=1)
        predictions = np.where(predictions > 255, 255, predictions)
        predictions = np.where(predictions < 0, 0, predictions)
        predicted_image = merge_blocks(predictions.astype(int), height, width)

        original_images.append(original_image)
        predicted_images.append(predicted_image)

        image_ssim.append(find_ssim(original_image, predicted_image, data_range=255, multichannel=True))
        image_psnr.append(find_psnr(original_image, predicted_image))
        image_num = image_num + 1
"""
for i in range(len(image_psnr)):
    print(image_psnr[i])
for i in range(len(image_psnr)):
    print(image_ssim[i])
"""
avg_psnr = sum(image_psnr)/image_num
avg_ssim = sum(image_ssim)/image_num
print("Average PSNR value of Kodak dataset: ", avg_psnr)
print("Average SSIM value of Kodak dataset: ", avg_ssim)

# Visualizing the original and the predicted images
index = 11

# Visualizing the original image
org_img = original_images[index].astype("uint8")
cv2_original_image = cv2.resize(org_img, (2*org_img.shape[1],2*org_img.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("Original Image", cv2_original_image)
cv2.waitKey(0)
cv2.imwrite("original_image_"+str(index)+".png", original_images[index])

# Visualizing the predicted image
pred_img = predicted_images[index].astype("uint8")
cv2_predicted_image = cv2.resize(pred_img, (2*pred_img.shape[1],2*pred_img.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("Predicted Image", cv2_predicted_image)
cv2.waitKey(0)
cv2.imwrite("bayer_predicted_image"+str(index)+".png", predicted_images[index])

print("\nImage PSNR: ", find_psnr(original_images[index], predicted_images[index]))
""" """