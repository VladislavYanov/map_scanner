import cv2
import numpy as np
import os
import time
import shutil

def compute_laplacian_var(image):
    laplacian = cv2.Laplacian(image, cv2.CV_8U)
    laplacian_var = np.var(laplacian)
    return laplacian_var

def get_laplist_from_vid(vid_path):
    laplist = []
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap = compute_laplacian_var(gray)
            laplist.append(lap)
        else:
            break
    cap.release()
    return laplist

def find_max_indices(arr, n):
    part_size = len(arr) // n
    max_indices = []

    for i in range(n):
        start_index = i * part_size
        end_index = (i + 1) * part_size
        part = arr[start_index:end_index]
        max_index = np.argmax(part)
        max_indices.append(max_index + start_index)

    return max_indices

def save_framelist(vid_path, ind_list, out_path, resize_perc = 100):
    cap = cv2.VideoCapture(vid_path)
    last = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    frame = cap.retrieve()[1]
    filename = os.path.join(out_path, str(1).zfill(6)+'.png')
    width = int(frame.shape[1] * resize_perc / 100)
    height = int(frame.shape[0] * resize_perc / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(filename, frame)
    for ind in ind_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        frame = cap.retrieve()[1]
        filename = os.path.join(out_path, str(ind).zfill(6)+'.png')
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename, frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last)
    frame = cap.retrieve()[1]
    filename = os.path.join(out_path, str(int(last)).zfill(6)+'.png')
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(filename, frame)

def save_best_frames_from_video(path_to_video, output_path, num_of_frames=8, resize_percent=50):
    os.makedirs(output_path,exist_ok=True)
    time1 = time.time()
    laplist = get_laplist_from_vid(path_to_video)
    time2 = time.time() - time1
    print('1/2 complited in:', time2)
    ind_list = find_max_indices(laplist, num_of_frames)
    time3 = time.time()
    save_framelist(path_to_video, ind_list, output_path, resize_percent)
    time4 = time.time() - time3
    print('complited in:', time4)




