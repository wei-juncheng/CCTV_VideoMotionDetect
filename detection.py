import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join 
import time
import threading

def MotionDetect(filename):

    # 讀取影片檔
    cap = cv2.VideoCapture(filename)

    # 初始化平均畫面
    ret, frame = cap.read()

    #調整影像Frame大小
    resize_y_end = 600
    resize_x_end = 550
    resize_y_start = 50
    resize_x_start =210
    frame = frame[resize_y_start:resize_y_end,resize_x_start:resize_x_end]

    avg = cv2.blur(frame, (2, 2))
    avg_float = np.float32(avg)

    # 計算影片中總共偵測到幾次物體移動
    MotionCounter = 0


    while(cap.isOpened()):
        # 讀取一幅影格
        ret, frame = cap.read()

        # 若讀取至影片結尾，則跳出
        if ret == False:
            break
        # 裁剪Frame尺寸，只專注在特定區域避免過多雜訊干擾
        frame = frame[resize_y_start:resize_y_end,resize_x_start:resize_x_end]

        # 模糊處理
        blur = cv2.blur(frame, (2, 2))
        
        # 計算目前影格與平均影像的差異值
        diff = cv2.absdiff(avg, blur)

        # 將圖片轉為灰階
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        
        # 篩選出變動程度大於門檻值的區域
        ret, thresh = cv2.threshold(gray, 6, 255, cv2.THRESH_BINARY)
        
        
        # 去除雜訊
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) #先腐蝕，後膨脹，去白噪點
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) #先膨脹，後腐蝕，去黑噪點

        # 判斷影像是否有任何移動點，計算整個影片移動點發生次數
        if np.any(thresh):
            MotionCounter+=1

        
        # 呈現影像
        cv2.imshow('thresh',thresh)
        cv2.imshow('origin',frame)

        #使影片接近正常速度呈現
        time.sleep(0.033)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 更新平均影像
        cv2.accumulateWeighted(blur, avg_float, 0.01)
        avg = cv2.convertScaleAbs(avg_float)
        
        

    print(filename,"偵測到",MotionCounter,"個變化")
    #關閉影像
    # cap.release()




def main():
    
    # 影片所在目錄的路徑(windows跟Linux的斜線轉換要注意)
    mypath = ".\\MotionVideo"

    # 取得該目錄下所有檔案與子目錄名稱
    files = listdir(mypath)

    #儲存所有MP4檔案名稱的List
    MP4FileNameList = []

    # 蒐集所有結尾是.mp4的檔案名稱
    for f in files:
        fullpath = join(mypath,f)
        # 判斷是否為mp4結尾:
        if isfile(fullpath) and fullpath.endswith('.mp4'):
            MP4FileNameList.append(fullpath)

    for index, videoFile in enumerate(MP4FileNameList):
        MotionDetect(videoFile)


if __name__ == '__main__':
    main()