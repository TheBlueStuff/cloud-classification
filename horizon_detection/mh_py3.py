import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3 
import matplotlib.pyplot as plt 
from PIL import Image
import cv2
import sys

def plotPixelData(old_path, scale):
    #opening and resizing an image in PIL
    img_file = Image.open(old_path)
    [xSize, ySize] = img_file.size
    img_file = img_file.resize((int(xSize/scale),int(ySize/scale)), Image.ANTIALIAS)
    img = img_file.load()

    #size of the new downsampled image
    [xSize, ySize] = img_file.size
    _r = []
    _g = []
    _b = []
    colours = []
    for x in range(0,xSize):
        for y in range(0,ySize):
            [r,g,b] = img[x,y]
            r /= 255.0
            _r.append(r)
            g /= 255.0
            _g.append(g)
            b /= 255.0
            _b.append(b)
            colours.append([r,g,b])
            
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(_r,_g,_b, c=colours, lw=0)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    fig.add_axes(ax)
    plt.show()
    return

def line(m, b, x, y):
    return y - m*x - b

def detectHorizon(cvImg, xSize, ySize):
    #keeping a resolution of 50 to generate the values of slope and y intercept
    #resolution can be changed by changing this value
    res = 50
    slope = np.linspace(-1,1,res)
    inter = np.linspace(0,ySize,res)
    maximum = []
    J_max = 0

    #iterate over all the slope and intercept values
    for m in range(len(slope)):
        for b in range(len(inter)):
            sky = [] #array of pixel values containing sky
            gnd = [] #array of pixel values containing ground

            #iterate over all the pixels in the image and add them to sky and gnd
            for i in range(xSize):
                for j in range(ySize):
                    if((line(slope[m],inter[b],i,j)*(-1*inter[b])) > 0):
                        sky.append(cvImg[j,i])
                    else:
                        gnd.append(cvImg[j,i])


            #find covariance of the sky and gnd pixels
            sky = np.transpose(sky)
            gnd = np.transpose(gnd)
            try:
                co_s = np.cov(sky)
                co_g = np.cov(gnd)
                co_sD = np.linalg.det(co_s)
                co_gD = np.linalg.det(co_g)
                eig_s, _ = np.linalg.eig(co_s)
                eig_g, _ = np.linalg.eig(co_g)

                J = 1/(co_sD + co_gD + (eig_s[0]+eig_s[1]+eig_s[2])**2 + (eig_g[0]+eig_g[1]+eig_g[2])**2)
                if J > J_max:
                    J_max = J
                    maximum = [slope[m], inter[b]]
                    print (maximum)
            except Exception:
                pass

    return maximum

def display(window, image):
    cv2.namedWindow( window ) 
    cv2.imshow( window, image );         
    cv2.waitKey(0) 


def plot(cvImg, horizon, path):
    xSize = cvImg.shape[1]
    ySize = cvImg.shape[0]
    print ("xSize", xSize)
    m = horizon[0]
    b = horizon[1]
    y2 = int(m*(xSize-1)+b)
    cv2.line(cvImg, (0,int(b)), (xSize-1, y2), (0,0,255), 2)
    display("horizon", cvImg)
    # Compute total area
    area = cvImg.shape[0] * cvImg.shape[1]
    # Compute area under line
    area_under_line = np.trapz([ySize - int(b), ySize - y2], x=[0, xSize-1])
    # Compute portion of image under line
    ratio = area_under_line/area
    print ("Horizon ratio", ratio)

def main():
    filename = sys.argv[1]
    filename = "sample images/" + filename
    #just some constants
    old_path = []
    # for i in range(1, 34):
    #   k = "sample images/image"+`i`+".png"
    #   print k
    #   old_path.append(k)
    old_path.append(filename)

    scale = 10

    for path in old_path:
        #opening the new image in CV
        print ("--------------------------------")
        print ("Processing image: ", path)
        cvImg_original = cv2.imread(path)
        # Crop watermarks
        cvImg_original = cvImg_original[50:995, :]
        ySize = cvImg_original.shape[0]
        xSize = cvImg_original.shape[1]
        cvImg = cv2.resize(cvImg_original, (0,0), fx=1/scale, fy=1/scale) 
        ySize = cvImg.shape[0]
        xSize = cvImg.shape[1]
        # cvImg = cv2.GaussianBlur(cvImg,(5,5),0)

        # plot pixel data in RGB space
        # plotPixelData(path, 10)

        horizon = []
        horizon = detectHorizon(cvImg, xSize, ySize)

        horizon[1] *= scale
        plot(cvImg_original, horizon, path)
        
if __name__ == '__main__':
    main()





