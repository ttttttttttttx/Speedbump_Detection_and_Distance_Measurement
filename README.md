# 减速带检测和测距

### 一、编译环境

#### 1. 创建编译环境

这里使用 Anaconda 创建用于运行 YOLOv8 训练模型的虚拟环境，如果没有 Anaconda 可以手动添加以下依赖项或使用python终端下载。

使用命令行，创建用于运行 YOLOv8 的虚拟环境并激活

```
conda create -n yolov8 python=3.8
conda activate yolov8
```

在激活的 yolov8 环境中，通过cd命令跳转到该项目 Detection-and-Distance-Measurement-of-Speedbumps 文件夹地址。

执行以下命令安装所需依赖项：

```
pip install ultralytics
```

其他所需的依赖项如opencv等，请根据需要添加。

#### 2. 在 pycharm 中使用上述 conda 虚拟环境

【设置】-【项目】-【Python 解释器】-【添加】-【Conda 环境】-【现有环境】：

解释器：上述创建的 yolov8 环境的地址，可能为 `(自己的下载地址)\anaconda3\envs\yolov8\python.exe`

Conda 可执行文件：`(自己的下载地址)\anaconda3\Scripts\conda.exe`

### 二、准备工作

#### 1. 相机标定

计算相机参数，去畸变

设备：鱼眼相机、棋盘格

#### 2. 单应矩阵标定

去畸变 -> 地面坐标系

设备：鱼眼相机、小车、地面关键点（瓷砖）

#### 3. 减速带检测模型训练

减速带检测

工具：yolov8（使用老师提供的数据集进行训练）

### 三、文件夹说明

- captured_video：直接拍摄的视频

- captured_undistorted_video：拍摄的视频和去畸变后的视频

- calibration_images：相机标定所用棋盘格图片

- calibration_corners_images：棋盘格角点检测图片

- marked_points_image：测距所用的地面瓷砖图片

- undistorted_frames：原始视频被拆分后的去畸变的帧图片

- marked_frames：标注了减速带和距离的去畸变的帧图片

- generated_video：最后生成的减速带测距视频

- configs：存储配置文件
  - H_matrix.npy：透视变换矩阵
  - camera_intrinsic.py：相机内参矩阵和畸变系数
  
- utils：存放一系列工具函数
  - capture.py：捕捉拍摄图片并保存
  - file_operations.py：文件操作（删除/写入文件）
  - generate_video.py：由视频帧生成完整视频
  - split_video_and_undistort.py：视频逐帧去畸变
  
- calibration.py：拍摄照片/相机标定/图片去畸变
  
- capture_video.py：拍摄视频（可同时生成去畸变视频）
  
- detect.py：实现减速带检测和测距功能，生成视频结果
  
- homography.py：通过地面瓷砖图片生成变换矩阵H
  
- speedbump.pt：模型文件
  
### 四、实现

#### 1. 图像去畸变

#### 2. 减速带检测

#### 3. 减速带距离计算

在 detect.py 中完成
