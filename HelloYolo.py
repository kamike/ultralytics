import cv2
from PIL import Image

from ultralytics import YOLO

imgFolder="C:\\Users\\Administrator\\Pictures\\testImg"

print("==============start===========")

# 加载 YOLO 模型，"model.pt" 是预训练模型的文件路径
model = YOLO("yolo11n.pt")
# 实时摄像头输入
# results = model.predict(source="0")

# 从文件夹读取图像
results = model.predict(source=imgFolder, show=True)


# 从 PIL 图像对象
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)

# 使用 OpenCV 打开一个图像文件，将其转换为 ndarray 对象进行预测。save=True 保存处理后的图像，save_txt=True 会将预测结果保存为文本文件格式（通常是标注文件）
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
# results = model.predict(source=[im1],show=True)

# 使用 cv2 显示每个预测结果
for img in results:
    cv2.imshow("Prediction", img.orig_img)  # 修改为 img.orig_img 根据实际接口

# 在这里设置等候时间为 0，表示无限等候，确保窗口保持打开
cv2.waitKey(0)

# 销毁所有窗口
cv2.destroyAllWindows()

print("==============complete===========")
