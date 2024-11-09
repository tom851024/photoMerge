import cv2
import numpy as np
# import matplotlib.pyplot as plt

# 定義滑鼠點擊事件的回調函數
def get_coordinates(event, x, y, flags, param):
    global click_count, points_img1, points_img2, current_image, image_copy, step  # 使用全域變數
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊事件
        click_count += 1  # 增加點擊計數
        print(f"點擊的座標為: x={x}, y={y}")
        # 在圖片上顯示點擊順序數字
        if current_image == 1:
            points_img1.append((x, y))  # 在第一張圖片上記錄點
            cv2.putText(image_copy, str(click_count), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_image == 2 and step == 1:
            points_img2.append((x, y))  # 在第二張圖片上記錄點
            cv2.putText(image_copy, str(click_count), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_image == 2 and step == 2:
            points_img2_2.append((x, y))  # 在第二張圖片上記錄點
            cv2.putText(image_copy, str(click_count), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_image == 3:
            points_img3.append((x, y))
            cv2.putText(image_copy, str(click_count), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 當三個點被記錄後，切換到下一張圖片或關閉視窗
        if click_count == 3:
            if current_image == 1:
                # 切換到第二張圖片
                current_image = 2
                image_copy = img2.copy()
                click_count = 0  # 重置點擊計數
            elif current_image == 2 and step == 1:
                print("第一組點選完成: img1 和 img2 對齊")
                step = 2
                current_image = 3
                image_copy = img3.copy()
                click_count = 0  # 重置點擊計數
            elif current_image == 3:
                print("第二組點選完成: img2 和 img3 對齊")
                current_image = 2
                image_copy = img2.copy()
                click_count = 0  # 重置點擊計數
            elif current_image == 2 and step == 2:
                # 結束並關閉視窗
                print("第一張圖片的三個點:", points_img1)
                print("第二張圖片的三個點:", points_img2)
                print("第三張圖片的三個點:", points_img3)

                # 計算 affine_matrix
                affine_matrix = calculate_affine_transform(points_img2, points_img1)
                affine_matrix2 = calculate_affine_transform(points_img3, points_img2_2)

                print("Affine Matrix:\n", affine_matrix)
                print("Affine Matrix2:\n", affine_matrix2)

                 # 合成圖片
                merge_images(affine_matrix, affine_matrix2)
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

def merge_images(affine_matrix, affine_matrix2):
    height, width, _ = img1.shape
    output_canvas = np.zeros((height * 2, width * 2, 3), dtype=np.float32)

    # Step 1: 將 img1 放置於畫布左上角
    img1_float = img1.astype(np.float32) / 255.0
    output_canvas[0:height, 0:width] = img1_float

    # Step 2: 仿射變換 img2 並直接疊加到畫布
    warped_img2 = cv2.warpAffine(img2, affine_matrix, (width * 2, height * 2))
    warped_img2 = warped_img2.astype(np.float32) / 255.0

    # 直接將 img2 疊加到畫布上
    output_canvas = np.maximum(output_canvas, warped_img2)

    # Step 3: 仿射變換 img3 並直接疊加到畫布
    warped_img3 = cv2.warpAffine(img3, affine_matrix2, (width * 2, height * 2))
    warped_img3 = warped_img3.astype(np.float32) / 255.0

    # 直接將 img3 疊加到畫布上
    output_canvas = np.maximum(output_canvas, warped_img3)

    # 正規化並顯示合成結果
    output_canvas = np.clip(output_canvas, 0, 1)
    output_canvas = (output_canvas * 255).astype(np.uint8)

    # 儲存合成圖片
    cv2.imwrite("merged_image.jpg", output_canvas)  # 以 JPEG 格式儲存圖片
    
    # 顯示結果
    cv2.imshow("Merged Image", output_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


click_count = 0
current_image = 1
points_img1 = [] #1號照片對應點
points_img2 = [] #1號照片和2號照片對應點
points_img2_2 = [] #2號照片和3號照片對應點
points_img3 = [] #3號照片對應點
step = 1

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

