import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import glob
import time
from detection_helpers import *


car_image_filenames = glob.glob('./vehicles/*/*.png')
num_car_images = len(car_image_filenames)
non_car_image_filenames = glob.glob('./non-vehicles/*/*.png')
num_non_car_images = len(non_car_image_filenames)
print('# car images:', num_car_images, '\n# non-car images:', num_non_car_images)

'''
# used for testing throughput
data_subset_size = 5000 # number of car and non-car images to use

# select random car and non-car examples for testing
car_examples = np.random.choice(car_image_filenames, data_subset_size, replace = False)
non_car_examples = np.random.choice(non_car_image_filenames, data_subset_size, replace = False)
'''

car_examples = car_image_filenames
non_car_examples = non_car_image_filenames
car_features = []
non_car_features = []

for file in car_examples:
    image = cv2.imread(file)
    car_features.append(extract_features(image, spatial_colorspace = spatial_colorspace, spatial_size = spatial_size,
                                    hist_bins = histogram_bins, hog_conversion = hog_conversion, channel = 1,
                                    orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block))
for file in non_car_examples:
    image = cv2.imread(file)
    non_car_features.append(extract_features(image, spatial_colorspace = spatial_colorspace, spatial_size = spatial_size,
                                    hist_bins = histogram_bins, hog_conversion = hog_conversion, channel = 1,
                                    orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block))

# put all examples together to be split later by train_test_split()
X = np.vstack((car_features, non_car_features)).astype(np.float64)
# fit per-column scaler
X_scaler = StandardScaler().fit(X)
# apply scaler
scaled_X = X_scaler.transform(X)

# labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
print(len(scaled_X))
print(len(y))
# divide selected data into training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = rand_state)

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

joblib.dump(svc, './vehicle_classifier.pkl')
joblib.dump(X_scaler, './X_scaler.pkl')