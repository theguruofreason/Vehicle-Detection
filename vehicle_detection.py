import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.feature import hog
import glob
from sklearn.externals import joblib
from detection_helpers import *


svc = joblib.load('./vehicle_classifier.pkl')
X_scaler = joblib.load('./X_scaler.pkl')
test_image_files = glob.glob('./test_images/*.jpg')
selection = np.random.choice(test_image_files, 1 )[0]
example = cv2.imread(selection)


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = extract_features(test_img, use_spatial = use_spatial, spatial_colorspace = spatial_colorspace,
                                         spatial_size = spatial_size, color_channels = color_channels,
                                         use_histogram = use_histogram, hist_bins = histogram_bins,
                                         use_hog = use_hog, hog_conversion = hog_conversion, hog_channels = hog_channels,
                                         orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform((np.array(features)).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 5)
    # Return the image
    return img

y_start_stop = [[376, 600], [404, 660], [420, 660]] # Min and max in y to search in slide_window()
x_start_stop = None
xy_overlap = [(0, 0), (0.5, 0.5), (.80, .80)]
hog_channel = 'ALL'
small_window = (32, 32)
medium_window = (64, 64)
large_window = (96, 96)
window_sizes = [small_window, medium_window, large_window]
windows = []
draw_image = np.copy(example)

for i in range(len(window_sizes)):
    windows.extend(slide_window(example, x_start_stop=[None, None], y_start_stop=y_start_stop[i],
                    xy_window=window_sizes[i], xy_overlap=xy_overlap[i]))
print(len(windows))

produce_images = False

if produce_images:
    hot_windows = search_windows(example, windows, svc, X_scaler, color_space=spatial_colorspace,
                                 spatial_size=spatial_size, hist_bins=histogram_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block)
    print(len(hot_windows))

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick = 5)

    cv2.imshow('test', window_img)
    cv2.waitKey(0)
    file_i = 1


    for file in test_image_files:
        image = cv2.imread(file)
        image_hot_windows = search_windows(image, windows, svc, X_scaler, color_space=spatial_colorspace,
                                           spatial_size=spatial_size, hist_bins=histogram_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block)
        boxes_image = draw_boxes(image, image_hot_windows, color = (0, 0, 255), thick = 5)
        cv2.imwrite('./output_images/test' + str(file_i) + '.png', boxes_image)
        print ('file', file_i, 'created')
        file_i += 1



from moviepy.editor import VideoFileClip

class VideoProcessor(object):
    def __init__(self, windows, frames_to_keep, heatmap_threshold):
        self.prevoius_hots = []
        self.frames_to_keep = frames_to_keep
        self.hot_window_acc = []
        self.windows = windows
        self.heatmap_threshold = heatmap_threshold

    def pipeline(self, frame):
        recolored = convert_color(frame, conv = 'RGB2BGR')
        heatmap = np.zeros_like(frame)
        draw_image = np.copy(frame)
        frame_hot_windows = search_windows(recolored, self.windows, svc, X_scaler, color_space=spatial_colorspace,
                        spatial_size=spatial_size, hist_bins=histogram_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block)
        if len(self.hot_window_acc) >= self.frames_to_keep:
            self.hot_window_acc.pop(0)
            self.hot_window_acc.append(frame_hot_windows)
        else:
            self.hot_window_acc.append(frame_hot_windows)
        for window_sets in self.hot_window_acc:
            add_heat(heatmap, window_sets)
        thresholded_heatmap = apply_threshold(heatmap, self.heatmap_threshold)
        labels = label(thresholded_heatmap)
        print(labels[1], 'cars found')
        if labels[1] != 0:
            result = draw_labeled_bboxes(draw_image, labels)
        else:
            result = draw_image
        return result


project_video_output = './project_output.mp4'
clip = VideoFileClip('./project_video.mp4')
first_frame = clip.get_frame(0)

my_video_processor = VideoProcessor(windows, frames_to_keep = 7, heatmap_threshold = 5)

pv_clip = clip.fl_image(my_video_processor.pipeline)
pv_clip.write_videofile(project_video_output, audio = False)