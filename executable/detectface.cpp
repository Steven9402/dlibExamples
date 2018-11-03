//
// Created by cuizhou on 18-3-19.
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dlibface/FaceDetection.h"

using namespace cv;
using namespace std;
using namespace dlib;

int main()
{
    /**
     * step1: imgdir
     * step2: parameter_path
     */
    char* imgdir="/home/cuizhou/myGitRepositories/dlibExamples/data/detection/2009_004587.jpg";
    char* parameter_path = "../../res/parameters_facedetection.ini";


    Mat srcImage = imread(imgdir);

    FaceDetection face_detection_impl(parameter_path);

    std::vector<dlib::rectangle> facerects = face_detection_impl.detectFace(srcImage);

    for(dlib::rectangle facerect:facerects){
        double left = facerect.left();
        auto top = facerect.top();
        auto right = facerect.right();
        auto bottom = facerect.bottom();

        cv::rectangle(srcImage,Point(left,top),Point(right,bottom),Scalar(0,255,255),1,8,0);
    }

    imshow("srcImage",srcImage);
    waitKey(0);

    return 1;
}
