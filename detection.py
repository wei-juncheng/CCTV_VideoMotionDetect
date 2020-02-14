import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join ,isdir
import time
import threading

def MotionDetect(filename):

    # 讀取影片檔
    cap = cv2.VideoCapture(filename)

    # 初始化平均畫面
    ret, frame = cap.read()

    # 設定特定區域進行觀察，程式只會偵測這個目標範圍內的動態，避免過多雜訊干擾結果
    # 若無需則都輸入0，程式會直接讀取影片的完整大小
    # 有些監視器會有時間標記，會被程式誤判為物體移動，因此強烈建議先設定目標區域
    resize_y_start = 35 # 左上角y座標
    resize_x_start =20 # 左上角x座標
    resize_y_end = 500 # 右下角y座標
    resize_x_end = 600 # 右下角x座標

    if resize_x_end != 0 and resize_y_end != 0:
        # 裁剪初始Frame尺寸，只專注在特定區域避免過多雜訊干擾
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
        
        frame_show = frame.copy()

        if resize_x_end != 0 and resize_y_end != 0:
            # 顯示關注區
            cv2.rectangle(frame_show, (resize_x_start, resize_y_start), (resize_x_end, resize_y_end), (0, 255, 0), 2)

            # 裁剪每個Frame尺寸，只專注在特定區域避免過多雜訊干擾
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
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) #先腐蝕，後膨脹
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) #先膨脹，後腐蝕

        # 判斷影像是否有任何移動點，計算整個影片移動點發生次數
        if np.any(thresh):
            MotionCounter+=1

        
        # 呈現影像
        cv2.imshow('origin',frame_show)
        cv2.imshow('thresh',thresh)

        #使影片接近正常速度呈現
        time.sleep(0.033)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 更新平均影像
        cv2.accumulateWeighted(blur, avg_float, 0.01)
        avg = cv2.convertScaleAbs(avg_float)
        
        

    # print(filename,"偵測到",MotionCounter,"個變化")
    print("{0}偵測到{1:3}個Frame有變化".format(filename, MotionCounter ))
    #關閉影像
    cap.release()




def main():
    
    # 影片所在目錄的路徑(windows跟Linux的斜線轉換要注意)
    input_path = ".\\MotionVideo"

    files = []
    try:
        if isfile(input_path):
            files.append(input_path)
        elif isdir(input_path):
            # 取得該目錄下所有檔案與子目錄名稱
            files = listdir(input_path)
        else:
            print('無法讀取檔案')
            return
    except:
        print("請輸入有效的檔案或資料夾名稱")
        return
    
    #儲存所有MP4檔案名稱的List
    MP4FileNameList = []

    # 蒐集所有結尾是.mp4的檔案名稱
    for f in files:
        fullpath = join(input_path,f)
        # 判斷是否為mp4結尾:
        if isfile(fullpath) and fullpath.endswith('.mp4'):
            MP4FileNameList.append(fullpath)

    for index, videoFile in enumerate(MP4FileNameList):
        MotionDetect(videoFile)
    

if __name__ == '__main__':
    main()