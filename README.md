##dlib face 使用demo
可执行文件包括：

###dlib自带demo
用dlib的数据格式读取图片
>face_detection_ex_oneimg 检测一张图片,返回人脸框
>face_landmark_detection_ex 检测一张图片,并提取68特征点,返回人脸框,和dlib的方式可视化结果
>dnn_face_recognition_ex 检测一张图片,并提取5特征点,旋转为正,计算128

###类封装和调用
使用opencv的Mat格式传递图片

库封装
>dlibfacedetection 包含 dlib::dlib --检测人脸

>dlibdeepfaceprocess 包含 dlibfacedetection dlib::dlib --人脸检测 + 并提取5特征点,旋转为正,计算128

>dlibdeepfacedescriptormanager 包含 dlibfacedetection dlibdeepfaceprocess dlib::dlib --集成上述所有功能 + 人脸featurepool的创建保存&比对功能

调用cpp
>detectface 检测一张图片,返回人脸框
>deepfeatureextration 检测一张图片,并提取5特征点,旋转为正,提取128特征

> pufa_CameraDetectandMakeFacePool 视频检测,保存人脸图片,人手移动到facepool文件夹中 
> pufa_EncodeFacePool 从facepool文件夹中读取图片,计算feature 存入 txt
> pufa_DetectFaceandVerify 从txt读取feature,或从facepool中计算feature,视频多人人脸验证 

todo
>增加人脸跟踪




