import cv2
import time
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from datetime import datetime


def load_resnet_model(model_path=None):
    """加载ResNet模型"""
    # 如果没有提供模型路径，使用预训练的ResNet18
    if model_path is None:
        model = torchvision.models.resnet18(pretrained=True)
        # 修改最后一层以适应分类
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 5)
    else:
        # 加载自定义模型
        model = torchvision.models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 5)
        model.load_state_dict(torch.load(model_path))

    model.eval()  # 设置为评估模式
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def predict_images(model, device, image_paths):
    """使用模型预测图片"""
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 准备图像
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = preprocess(img)
        images.append(img_tensor)

    # 批量预测
    with torch.no_grad():  # 不计算梯度，节省内存
        inputs = torch.stack(images).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    return preds.cpu().numpy()  # 返回预测结果


def capture_photos(save_dir="captured_photos", duration=10, batch_size=10,
                   model_path=None, zero_threshold=0.5):
    """
    调用摄像头并保存照片，每收集batch_size张照片进行一次预测
    :param save_dir: 照片保存目录
    :param duration: 拍摄持续时间（秒）
    :param batch_size: 每多少张照片进行一次预测
    :param model_path: 模型路径，None则使用预训练模型
    :param zero_threshold: 判断为"正确"的0结果比例阈值
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    print("加载ResNet模型...")
    model, device = load_resnet_model(model_path)
    print(f"使用设备: {device}")

    # 初始化摄像头（0表示默认摄像头，多个摄像头可尝试1,2...）
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 目标帧率：每秒1张
    target_fps = 10
    interval = 1.0 / target_fps  # 每张照片的时间间隔（秒）

    start_time = time.time()
    frame_count = 0
    batch_images = []  # 存储当前批次的图片路径

    print(f"开始拍摄，将持续{duration}秒，每秒保存{target_fps}张照片...")
    print(f"每{batch_size}张照片进行一次预测")
    print(f"照片将保存至：{os.path.abspath(save_dir)}")
    print("按 'q' 键可提前退出")

    try:
        while time.time() - start_time < duration:
            # 记录当前帧开始时间
            frame_start = time.time()

            # 捕获一帧图像
            ret, frame = cap.read()

            # 检查帧是否捕获成功
            if not ret:
                print("无法获取图像帧，退出")
                break

            # 生成文件名（时间戳+序号，避免重复）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳
            filename = f"{save_dir}/frame_{timestamp}_{frame_count}.jpg"

            # 保存图像
            cv2.imwrite(filename, frame)
            frame_count += 1
            batch_images.append(filename)
            # print(f"已保存 {frame_count} 张照片")

            # 当收集到足够数量的图片时进行预测
            if len(batch_images) >= batch_size:
                preds = predict_images(model, device, batch_images)
                print(preds)

                # 统计结果
                zero_count = np.sum(preds == 0)
                zero_ratio = zero_count / len(preds)

                # 判断是否符合条件
                if zero_ratio < zero_threshold:
                    print(f"请直视摄像头")

                # 清空批次列表
                batch_images = []

            # 显示实时图像
            cv2.imshow("Camera Feed (Press 'q' to quit)", frame)

            # 控制帧率：如果处理过快，等待剩余时间
            elapsed = time.time() - frame_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

            # 检测按键，按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户手动退出")
                break

        # 处理剩余不足一个批次的图片
        if len(batch_images) > 0:
            print(f"对剩余{len(batch_images)}张图片进行预测...")
            preds = predict_images(model, device, batch_images)

            zero_count = np.sum(preds == 0)
            zero_ratio = zero_count / len(preds)
            print(f"预测结果中0的比例: {zero_ratio:.2f} ({zero_count}/{len(preds)})")

            if zero_ratio >= zero_threshold:
                print(f"判断结果: 符合条件 (0的比例超过{zero_threshold})")
            else:
                print(f"判断结果: 不符合条件 (0的比例低于{zero_threshold})")

    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print(f"拍摄结束，共保存 {frame_count} 张照片")


if __name__ == "__main__":
    # 可自定义参数
    capture_photos(
        save_dir="my_captures",
        duration=30,  # 拍摄30秒
        batch_size=10,  # 每10张照片预测一次
        model_path='best_model_res.pth',  # 可以替换为你的模型路径
        zero_threshold=0.5  # 0的比例超过50%则认为正确
    )
