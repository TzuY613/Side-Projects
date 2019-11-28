from cv2 import cv2 
import numpy as np
from matplotlib import pyplot as plt


imgName = input("name : ")
imgName = "FPK_"+ str(imgName) + ".jpg" 
img = cv2.imread(imgName,0)
#img = cv2.medianBlur(img,5) #模糊化
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
"""
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
"""

#cv2.imshow(imgName+"Original Image", img)
cv2.imwrite(imgName+"_Original_Image.png", img)
#cv2.waitKey(0)
#cv2.imshow(imgName+"Global Thresholding (v = 127)", th1)
cv2.imwrite(imgName+"_Global_Thresholding.png", th1)
#cv2.waitKey(0)
#cv2.imshow(imgName+"Adaptive Mean Thresholding", th2)
cv2.imwrite(imgName+"_Adaptive_Mean_Thresholding.png", th2)
#cv2.waitKey(0)
#cv2.imshow(imgName+"Adaptive Gaussian Thresholding", th3)
cv2.imwrite(imgName+"_Adaptive_Gaussian_Thresholding.png", th3)
#cv2.waitKey(0)

th1 = cv2.threshold(th1, 240, 255, 1)[1]  # ensure binary
th2 = cv2.threshold(th2, 240, 255, 1)[1]  # ensure binary
th3 = cv2.threshold(th3, 240, 255, 1)[1]  # ensure binary

labels = cv2.connectedComponentsWithStats(th1)
labels2 = cv2.connectedComponentsWithStats(th2)
labels3 = cv2.connectedComponentsWithStats(th3)
print(labels3)

def imshow_components(labels,num):
    # Map component labels to hue val
    #print(labels)
    label_hue = np.uint8(50000000*labels/np.max(labels))
    #print(label_hue)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to white (把背景用白色顯示)
    labeled_img[label_hue==0] = 255

    #cv2.imshow(imgName+'labeled.png', labeled_img)
    cv2.waitKey()
    if num == 1 :
        cv2.imwrite(imgName+'_labeled_adaptive_global.png', labeled_img)
    if num == 2 :
        cv2.imwrite(imgName+'_labeled_adaptive_mean.png', labeled_img)
    if num == 3 :
        cv2.imwrite(imgName+'_labeled_adaptive_gaussian.png', labeled_img)
    return labeled_img
        

imshow_components(labels[1],1)
imshow_components(labels2[1],2)
labelImg = imshow_components(labels3[1],3)
cv2.imwrite(imgName+'_labeled_adaptive_gaussian.png', labelImg)


def smooth(countList) :
    nearList = []
    newList = []
    for i in range(0, len(countList)) :
        nearList.clear()
        for k in range(i-3, i+3) :
            if (k >= 0 and k < len(countList) and k != i) :
                nearList.append(countList[k])

        newList.append(int(np.median(nearList)))
        #countList[i] = int(np.median(nearList))

    return newList

def strongSmooth(countList) :
    nearList = []
    newList = []
    for i in range(0, len(countList)) :
        nearList.clear()
        for k in range(i-8, i+8) :
            if (k >= 0 and k < len(countList) and k != i) :
                nearList.append(countList[k])

        newList.append(int(np.median(nearList)))
        #countList[i] = int(np.median(nearList))

    return newList




def countComponent(labels) :
    count = 0
    lineCountList = []  # 紀錄每行component的數量
    componentList = []  # 該行component編號的list
    isFind = False
    img = np.zeros((labels.shape[0], labels.shape[1]))  # 建立矩陣(跟二維陣列不太一樣)
    #img = [([0]*labels.shape[0]) for i in range(labels.shape[1])]  # 建立二維陣列
    
    # dim = (shape[0], shape[1])
    # 畫出所有橫的切割線
    for i in range(0, labels.shape[0]) :  # line
        count = 0
        componentList.clear()
        for k in range(0, labels.shape[1]) :  # pixels in line
            isFind = False
            if (labels[i][k] != 0) :
                for j in range(0, len(componentList)) :
                    if(labels[i][k] == componentList[j]) :  # 該行已經有這個component存在
                        isFind = True
                if (not isFind) :
                    componentList.append(labels[i][k])
                    count+=1
        lineCountList.append(count)
        #print(count)
    
    lineCountList = smooth(lineCountList)
    #lineCountList = strongSmooth(lineCountList)
    setList = set(lineCountList)
    newList = list(setList)
    print(newList)
    print(np.percentile(lineCountList, 75))

    mean = np.mean(newList)
    median = np.median(newList)
    num = np.percentile(newList, 75)
    for i in range(0, img.shape[0]) :
        if (lineCountList[i] < median) :
            for k in range(0, img.shape[1]) :
                img[i][k] = 1
        else :
            for k in range(0, img.shape[1]) :
                img[i][k] = 0
        '''
        if (img[i][0] == 1) :
            print("1")
        else :
            print("0")
        '''
    drawLineChart(lineCountList)
    lineCountList.clear()
    # 畫出所有直的切割線
    for i in range(0, labels.shape[1]) :  # line
        count = 0
        componentList.clear()
        for k in range(0, labels.shape[0]) :  # pixels in line
            isFind = False
            if (labels[k][i] != 0) :
                for j in range(0, len(componentList)) :
                    if(labels[k][i] == componentList[j]) :
                        isFind = True
                if (not isFind) :
                    componentList.append(labels[k][i])
                    count+=1
        lineCountList.append(count)
        #print(count)

    lineCountList = smooth(lineCountList)
    #lineCountList = strongSmooth(lineCountList)
    setList = set(lineCountList)
    newList = list(setList)
    print(newList)
    print(np.percentile(newList, 75))

    mean = np.mean(newList)
    median = np.median(newList)
    num = np.percentile(newList, 75)
    for i in range(0, img.shape[1]) :
        if (lineCountList[i] < median) :
            for k in range(0, img.shape[0]) :
                img[k][i] = 1
        '''
        if (img[0][i] == 1) :
            print("1")
        else :
            print("0")
        '''
    drawLineChart(lineCountList)
    return img

def drawLineChart(y) :
    x = []
    for i in range(0, len(y)) :
        x.append(i+1)

    plt.scatter(x, y,alpha = 0.6)
    plt.xlabel("line number")
    plt.ylabel("component count")
    plt.title("each line's components")
    plt.show()

lineLabels = countComponent(labels3[1])
newImg= imshow_components(lineLabels,3)
#print(newImg.shape, img.shape)
alpha = 0.7
beta = 1-alpha
gamma = 0
img_add = cv2.addWeighted(labelImg, alpha, newImg, beta, gamma)

cv2.imshow('img_add',img_add)
cv2.imwrite(imgName+'_labeled_add.png', img_add)
cv2.waitKey()