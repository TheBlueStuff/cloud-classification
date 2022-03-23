from myHorizon import *
import os
import copy
import argparse
import glob
import time
# Hiperparameters
scale = 10
slide_x = 292
slide_y = 195
cropped_size = 750
lower_bound = 0.2
upper_bound = 0.45
first_hour = '0630'
last_hour = '1800'

RESULTS_PATH = 'data_base_final'

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='')
    args = parser.parse_args()
    folder = args.folder
    if folder == '':
        print("No folder provided")
        return
    folder = os.path.join(folder, "images", "*.jpg")
    for file in glob.glob(folder):
        info = os.path.basename(file)
        hour = info[7:-8]
        date_time = info[1:-8]
        if hour < first_hour or hour>last_hour:
            continue
        img = cv2.imread(file)
        # Remove watermarks
        img = img[50:995, :]
        cropped_imgs = []
        horizons = []
        init_y = 0
        # Slide
        for _ in range(2):
            init_x = 0
            for _ in range(4):
                cropped = img[init_y:init_y+cropped_size, init_x:init_x+cropped_size]
                init_x += slide_x
                cropped_imgs.append(cropped)
                ySize = cropped.shape[0]
                xSize = cropped.shape[1]
                cvImg = cv2.resize(cropped, (0,0), fx=1/scale, fy=1/scale) 
                ySize = cvImg.shape[0]
                xSize = cvImg.shape[1]
                horizon = detectHorizon(cvImg, xSize, ySize)
                horizon[1] *= scale
                horizons.append(horizon)
            init_y += slide_y
        ix = 0
        for img, horizon in zip(cropped_imgs, horizons):
            cache = copy.deepcopy(img)
            xSize = img.shape[1]
            ySize = img.shape[0]
            m = horizon[0]
            b = horizon[1]
            y2 = int(m*(xSize-1)+b)
            cv2.line(cache, (0,int(b)), (xSize-1, y2), (0,0,255), 2)
            area = img.shape[0] * img.shape[1]
            # Compute area under line
            area_under_line = np.trapz([ySize - int(b), ySize - y2], x=[0, xSize-1])
            # Compute portion of image under line
            ratio = area_under_line/area
            file_name = date_time + '_' + str(ix) + ".jpg"
            folder_name = date_time
            file_name = os.path.join(RESULTS_PATH, folder_name, file_name)
            if ratio >= lower_bound and ratio <= upper_bound:
                cv2.imwrite(file_name, img)
                ix = ix + 1

            print ("Horizon ratio", ratio)

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()