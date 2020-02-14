# CCTV Vedio Analyser (Motion Detection)
## 介紹
- 夜間的監視器畫面往往都是沒有人在走動的，但萬一有人或生物晚上偷偷闖入，可能會造成一些危害。 為了要揪出這些不速之客，許多人會選擇仔細的把每段影片看完，怕錯過一分一秒會錯過關鍵證據，這樣需要花費非常多時間還有精神，因此這個Python腳本可以幫你分析影片中是否有出現物體移動的跡象，並且透過OpenCV進行影像處理將移動前景以白色表示，固定背景則為黑色，並且統計出影片中總共有幾張Freme出現移動跡象，你可以在自己手動確認是否為監視器雜訊或是蚊蟲飛過產生的異常訊號。
## 說明
- 本工具適合針對監視器產生的許多小檔案進行分析，不適合針對長時間連續錄影進行分析。
- 若監視器產生出來的影片是連續且長時間的影像，請先使用其他工具進行分割，分割出來每個片段長度以1分鐘為最佳。
- 本專案分成一般python程式以及多執行序版本，主要影像處理邏輯相同，一般版本可以顯示目標區域並且依照有無偵測到動態改變外框顏色，多執行緒版本執行時間比較短~
- ``.\MotionVideo``資料夾中附上一段Demo用的影片，場景是監視器在偵測晚上老鼠出現的動作。
- 使用前請先更改``input_path``檔案路徑，可輸入單一檔案名稱或資料夾路徑

## 環境
- Python 3.6
- OpenCV (opencv-python)

## 使用
1. 請先下載本專案。 Clone this project first.
```
$ git clone https://github.com/wei-juncheng/CCTV_VideoMotionDetect.git
```
2. 使用pip安裝OpenCV以及其他必要套件。 Use pip to install OpenCV and other require packages.
```
pip install -r requirements.txt
```
3. 執行程式。 Luanch the python script. 

- 一般: ``$ python detection.py``
- 多執行緒: ``$ python detection_multithread.py``


## 參考資料
- https://blog.gtwang.org/programming/opencv-motion-detection-and-tracking-tutorial/