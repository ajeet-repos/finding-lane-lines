import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_line(m, c, height, width, img):
    # bottom point
    y1 = height
    x1 = (y1 - c) / m
    #print(x1, y1)
    
    y2 = height / 2 + constant_y
    x2 = (y2 - c) / m
    #print(x2, y2)
    if  math.isnan(x1) or math.isnan(x2) or math.isnan(y1) or math.isnan(y2):
        return
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=[255, 0, 0], thickness = 10)
    
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    ls = []
    rs = []
    linter = []
    rinter = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope < 0: #left line, negative slope
                ls += [slope]
                linter += [(y1-slope*x1)]
                linter += [(y2-slope*x2)]
            elif slope > 0: #right line, positive slop
                rs += [slope]
                rinter += [(y1-slope*x1)]
                rinter += [(y2-slope*x2)]
    
        
    global left_slope, right_slope, left_intercept, right_intercept
    
    # right slope
    rmean = np.mean(rs)
    # left slope
    lmean = np.mean(ls)
    #print(lmean, rmean)
    
    lcmean = np.mean(linter)
    rcmean = np.mean(rinter)
    
    height, width = img.shape[0], img.shape[1]
    
    left_slope += [lmean]
    left_intercept += [lcmean]
    right_slope += [rmean]
    right_intercept += [rcmean]
    
    #print(len(lines))
    #print(np.mean(left_slope), np.mean(right_slope))
    draw_line(np.mean(left_slope), np.mean(left_intercept), height, width, img)
    draw_line(np.mean(right_slope), np.mean(right_intercept), height, width, img)
    '''    
    draw_line(lmean, lcmean, height, width, img)
    draw_line(rmean, rcmean, height, width, img)
    '''
    #cv2.line(img, rightlines[0], rightlines[len(rightlines)-1], color, 30)
    
    #print(leftlines[len(leftlines)-1])
    #print(leftlines[0])
    #cv2.line(img, leftlines[0], leftlines[len(leftlines)-1], color, 100)
            
    
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=4)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# reading the test image
image_1 = mpimage.imread('test_images/solidWhiteCurve.jpg')
image_2 = mpimage.imread('test_images/solidWhiteRight.jpg')
image_3 = mpimage.imread('test_images/solidYellowCurve.jpg')
image_4 = mpimage.imread('test_images/solidYellowCurve2.jpg')
image_5 = mpimage.imread('test_images/solidYellowLeft.jpg')
image_6 = mpimage.imread('test_images/whiteCarLaneSwitch.jpg')



# now code for detecting lane lines in a video
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    #image_ = mpimage.imread(image)
    # 1. change the image to grayscale
    gray_image = grayscale(image)
    # 2. run the gaussian blur on it to smoothen the image
    kernel_size = 7
    blur_image = gaussian_blur(gray_image, kernel_size)
    # 3. run canny edge detection to detect the image
    low_threshold = 50
    high_threshold = 100
    canny_img = canny(blur_image, low_threshold, high_threshold)
    # 4. run the hough transform to get lines from the image obtained from canny edge detections
    
    # 5. create the masking area for the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    points = np.array([[0,ysize], [xsize/2 - 30,ysize/2 + constant_y], [xsize/2 + 30,ysize/2 + constant_y], [xsize, ysize]], dtype=np.int32)
    points = [points]
    mask_img = region_of_interest(canny_img, points)
    
    row = 1
    theta = math.pi / 180
    threshold = 30
    min_lin_length = 35
    max_line_gap = 100
    hough_lines_img = hough_lines(mask_img, row, theta, threshold, min_lin_length, max_line_gap)
    
    # 6. impose the both the images and print the result
    wighted_img = weighted_img(image, hough_lines_img)
    plt.imshow(wighted_img)

    return wighted_img

    

#process_image(image_1)
constant_y = 60
left_slope = []
left_intercept = []
right_slope = []
right_intercept = []
white_output = 'yellow_spyder.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


constant_y = 60
left_slope = []
left_intercept = []
right_slope = []
right_intercept = []
white_output = 'white_spyder.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

constant_y = 60
left_slope = []
left_intercept = []
right_slope = []
right_intercept = []
white_output = 'challenge_spyder.mp4'
clip1 = VideoFileClip("challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
