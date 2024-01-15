# imports
from preprocessing.HuangsFuzzyThresholding import Huang as hft
from preprocessing.PectoralMuscleRemoval import *
from preprocessing.RollingBallAlgorithm import *
from dataset_DDSM import Cancer_Classification_Data as ccd 


# morphological transformations function
def morph_trans(binarized_image, rolled_image):
    
    #erosion
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(binarized_image, kernel, iterations = 1)
    #dilation
    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    #merging
    merged = cv2.bitwise_and(dilation, rolled_image)
    
    return merged



# preprocessing function
def preprocessing(image):
    
    #convert to 8-bit grayscale
    img8 = (image/256).astype('uint8')
    grey_image = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
      
    
    #image resize
    img_resize = cv2.resize(grey_image, (320,320))
    
    #create ball
    ball = RollingBall(5)
    
    #create light background
    light_img = rolling_ball_background(img_resize)

    #apply rolling ball
    rolled_img = roll_ball(ball, light_img)
    
    
    #thresholding
    threshold = hft(rolled_img)
    
    #applying threshold to create binarized image
    binarized_img = cv2.threshold(rolled_img, threshold, cv2.THRESH_BINARY)
    
    
    #morphological transformations
    trans_img = morph_trans(binarized_img, rolled_img)
    
    
    ##pectoral muscle removal
    #flip
    flipped_img = right_orient_mammogram(trans_img)
    
    #resize 2
    img_resize2 = cv2.resize(flipped_img, (256,256))
    
    #canny edge
    canny_img = apply_canny(img_resize2)
    
    #hough lines
    lines = get_hough_lines(canny_img)
    sl_lines = shortlist_lines(lines)
    
    #removing pectoral
    rr, cc = remove_pectoral(sl_lines)
    trans_img[rr, cc] = 0
    
    return trans_img
