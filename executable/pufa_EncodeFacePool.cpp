//
// Created by cuizhou on 18-3-19.
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlibface/faceDescriptorManager.h>
#include "myfUtils/FileOperator.h"

using namespace cv;
using namespace std;
using namespace dlib;

int main()
{
    /**
     * 用于生成face_encodes.txt,保存人脸库128特征
     * step0:  ../data/facepool 放置人脸库
     * step1: 设置 parameters.ini init_face_lib=1
     * step2: 设置 parameter_path
     */

    char* parameter_dir = "../../res/parameters.ini";

    faceDescriptorManager face_descriptor_manager(parameter_dir);

    //faceDescriptorManager的析构函数中会将face encode 保存到txt中

    return 1;
}
