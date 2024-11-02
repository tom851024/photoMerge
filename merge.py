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

                # 計算 affine_matrix
                affine_matrix = calculate_affine_transform(points_img2, points_img1)
                print("Affine Matrix:\n", affine_matrix)

                # 計算畫布大小
                height, width, _ = img1.shape
                output_canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

                # 將目標圖像放在畫布的中心
                output_canvas[0:height, 0:width] = img1

                # 將來源圖像應用仿射變換並放置在畫布上
                warped_src_img = cv2.warpAffine(img2, affine_matrix, (width * 2, height * 2))
                output_canvas = cv2.addWeighted(output_canvas, 1, warped_src_img, 1, 0)

                # 顯示結果
                cv2.imshow("Merged Image", output_canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit()

def calculate_affine_transform(src_points, dst_points):
    A = []
    B = []
    for i in range(3):
        x_src, y_src = src_points[i]
        x_dst, y_dst = dst_points[i]
        A.append([x_src, y_src, 1, 0, 0, 0])
        A.append([0, 0, 0, x_src, y_src, 1])
        B.append(x_dst)
        B.append(y_dst)

    # 求解方程組以獲取仿射變換參數
    A = np.array(A)
    B = np.array(B)
    affine_params = np.linalg.lstsq(A, B, rcond=None)[0]

    # 組裝成 2x3 仿射矩陣
    affine_matrix = np.array([
        [affine_params[0], affine_params[1], affine_params[2]],
        [affine_params[3], affine_params[4], affine_params[5]]
    ])
    return affine_matrix


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

