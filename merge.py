import cv2
import numpy as np
# import matplotlib.pyplot as plt

# 定義滑鼠點擊事件的回調函數
def get_coordinates(event, x, y, flags, param):
    global click_count, points_img1, points_img2, current_image, image_copy  # 使用全域變數
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊事件
        click_count += 1  # 增加點擊計數
        print(f"點擊的座標為: x={x}, y={y}")
        # 在圖片上顯示點擊順序數字
        if current_image == 1:
            points_img1.append((x, y))  # 在第一張圖片上記錄點
            cv2.putText(image_copy, str(click_count), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_image == 2:
            points_img2.append((x, y))  # 在第二張圖片上記錄點
            cv2.putText(image_copy, str(click_count), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 當三個點被記錄後，切換到下一張圖片或關閉視窗
        if click_count == 3:
            if current_image == 1:
                # 切換到第二張圖片
                current_image = 2
                image_copy = img2.copy()
                click_count = 0  # 重置點擊計數
            elif current_image == 2:
                # 結束並關閉視窗
                print("第一張圖片的三個點:", points_img1)
                print("第二張圖片的三個點:", points_img2)
                cv2.destroyAllWindows()
                exit()

click_count = 0
current_image = 1
points_img1 = [] #1號照片對應點
points_img2 = [] #2號照片對應點

img1 = cv2.imread("1.jpg")
img2 = cv2.imread("2.jpg")
img3 = cv2.imread("3.jpg")

image_copy = img1.copy()


# 設置視窗名稱並綁定滑鼠事件
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_coordinates)

# 顯示圖片
while True:
    cv2.imshow('Image', image_copy)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 Esc 鍵退出
        break

cv2.destroyAllWindows()

