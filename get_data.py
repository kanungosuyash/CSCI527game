import time
import cv2
from PIL import ImageGrab
import numpy as np
from grabscreen import grab_screen
from getkeys import key_check
import sys
import os

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def key_label(keys) :

    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def save_data(output_file,data):

    if not os.path.exists(output_file):
        print("Saving data to new file")
    with open(output_file,'wb'):
        np.save(output_file,data)


def main():

    output_dir = sys.argv[1]
    file_num = 1
    frame_count = 0
    training_data = []
    paused = False
    print("Starting in")
    for i in list(range(5))[::-1] :
        print(i)
        time.sleep(1)

    while True:
        start = time.time()
        if not paused:
            img = grab_screen((0,0,800,600))
            img = cv2.resize(img,(400,300))
            keys = key_check()
            key_output = key_label(keys)
            training_data.append([img,key_output])
            frame_count += 1
            if len(training_data) % 100 == 0:
                print(f"{frame_count} Total Frames Collected")
                if len(training_data) % 500 == 0:
                    print('Saving data')
                    output_file = os.path.join(output_dir,f"training_data-{file_num}.npy")
                    save_data(output_file,training_data)
                    training_data = []
            # cv2.imshow("frame", img)
            end = time.time()
            print(f"Frame took {end-start} seconds")
            cv2.waitKey(100)
            if cv2.waitKey(1) & 0Xff == ord('q'):
                break

        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == '__main__' :

    main()