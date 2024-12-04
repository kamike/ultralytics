import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# 设置图像文件夹路径
imgFolder = "C:\\Users\\Administrator\\Pictures\\testImg"

print("==============start===========")

# 加载 YOLO 模型，"yolo11n-pose.pt" 是预训练模型的文件路径
model = YOLO("yolo11n-pose.pt")

# 打开图像
im1 = Image.open(imgFolder + "\\test4.jpg")
# 将 PIL 图像转换为 NumPy 数组
im1_cv = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)

# 获取预测结果
results = model.predict(source=im1)

# 创建一张新的图像用于绘制
output_image = im1_cv.copy()

# 遍历每个检测结果
for result in results:
    # 获取姿势关键点
    keypoints = result.keypoints  # 假设 keypoints 是一个 (1, 17, 3) 的数组，表示 1 个人体的 17 个关键点的 (x, y, confidence)

    if keypoints is None:
        print("未解析到数据")
        break

    print(keypoints.data)

    # 获取姿势关键点数据
    keypoints = keypoints.data.numpy()  # 转换为 NumPy 数组
    for person in keypoints:  # 遍历每个人
        for kp in person:  # 遍历每个关键点
            x, y, confidence = kp
            if confidence > 0.5:  # 仅绘制置信度大于阈值的关键点
                cv2.circle(output_image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
# 显示结果
cv2.imshow("Pose Estimation", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
output_path = imgFolder + "\\test1_pose.jpg"
cv2.imwrite(output_path, output_image)
