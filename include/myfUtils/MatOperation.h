//
// Created by cuizhou on 18-2-24.
//

#ifndef MSER_MYMATOPERATION_H
#define MSER_MYMATOPERATION_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

namespace myf{
    cv::Mat composeGrayVisMat(std::vector<Mat> matpool);

    bool expandRoi(int expandwidth,const cv::Rect& roi,cv::Rect& expandedROI,int matwidth,int matheight);
}

#endif //MSER_MYMATOPERATION_H
