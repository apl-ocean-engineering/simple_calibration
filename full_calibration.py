#!/usr/bin/env python3

import numpy as np
import cv2
import glob
import os
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

# This script is made to be placed in a directory containing folders of calibration images
# where each folder contains images taken at a specific focus value.
# Camera calibration will be run on the images within each folder, for the purpose
# of comparing the resulting intrinsic matrices and distortion vectors across different focal lengths
#
# Each folder should be named the focus value of the images conatined within it


# Create dict of focus values mapped to corresponding calibration images
folders = os.listdir(".")
imagefiles = None

scale_percent = 30 # change if want to resize image for faster processing
redraw = True

for path in os.listdir(os.getcwd()):
    imagefiles = glob.glob('**/*.png')

for x in folders:
    try:
        float(x)
    except ValueError:
        folders.remove(x)

focus_dict = defaultdict()
for focus in folders:
    focus_dict[focus] = []
    for file in imagefiles:
        if file.startswith(focus):
            focus_dict[focus].append(file)

# Set up variables to be filled in by calibrations
# Results are stored in a 4 x n matrix, where n = number of focus values
# TODO: reshape to n x 4 matrix?
# Index     Value
# 0         focus
# 1         fx
# 2         fy
# 3         norm of distortion vector
result_data = np.zeros(shape = (4, len(focus_dict)))

# Go through camera calibration in each focus and save to results
# (currently unsorted)

# termination criteria
# TODO: look into how this impacts calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

current_index = 0
for focus in focus_dict:
    for fname in focus_dict.get(focus):
        img = cv2.imread(fname)

        # resize image to help with computation
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        print("WORKING ON " + fname)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6,8),None)

        print("CORNERS FOUND FOR " + fname)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners

            # print("DREW CORNERS ON " + fname)

            # img = cv2.drawChessboardCorners(resized, (8,6), corners2, ret)
            # cv2.imshow('img',resized)
            # cv2.waitKey(0)

        lastname = fname # if desired, can redraw the last picture in the set with the found calibration

    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    except:
        print("error")

    # should now have results for this specific focus

    dist_norm = np.linalg.norm(dist)

    print("IN FOCUS = " + focus)
    print("FOUND FX = " + str(mtx[0,0]))
    print("FOUND FY = " + str(mtx[1,1]))
    print("FOUND DIST NORM = " + str(dist_norm))
    result_data[0, current_index] = focus
    result_data[1, current_index] = mtx[0,0]
    result_data[2, current_index] = mtx[1,1]
    result_data[3, current_index] = dist_norm

    current_index = current_index + 1

    if (redraw):
        img = cv2.imread(lastname)

        # make sure to scale image same as previously
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

        h,  w = gray.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort using found calibration
        dst = cv2.undistort(gray, mtx, dist, None, newcameramtx)

        print("REDRAW COMPLETE, SAVING")
        filename = str(focus) + ' redrawn.png'
        cv2.imwrite(filename,dst)


np.savetxt("full_calibration_result.csv", result_data, delimiter=",")


focuses = result_data[0]
fx_values = result_data[1]
fy_values = result_data[2]
norm_values = result_data[3]
# focuses serves as x values for these plots
#plt.plot(x, y, color, label)
# time to plot stuff!!
fig1 = plt.figure()
fig1.subplots_adjust(hspace = 1.0)

plt.subplot(3, 1, 1)
# plt.plot(focuses, fx_values,   color = 'r', label = 'fx')
plt.title('fx values')
plt.xlabel('focus values')
plt.scatter(focuses, fx_values, c = 'r')

plt.subplot(3, 1, 2)
# plt.plot(focuses, fy_values,   color = 'b', label = 'fy')
plt.title('fy values')
plt.xlabel('focus values')
plt.scatter(focuses, fy_values, c = 'b')

plt.subplot(3, 1, 3)
# plt.plot(focuses, norm_values, color = 'g', label = 'distortion vector')
plt.title('distortion vectors')
plt.xlabel('focus values')
plt.scatter(focuses, norm_values, color = 'g')
plt.show()
