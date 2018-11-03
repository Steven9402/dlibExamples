//
// Created by kyxu on 17-11-6.
//

#ifndef FACE_RECOGNITION_FACEDETECTION_H
#define FACE_RECOGNITION_FACEDETECTION_H

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/clustering.h>
#include <vector>
#include <string>


using namespace dlib;


// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

class FaceDetection {

public:
    FaceDetection();
    FaceDetection(std::string parameter_paths);

//    void setDetectThresh(float detect_thresh);
//    void setDetectModel(std::string model_path);

    std::vector<rectangle> detectFace(cv::Mat &image);

    std::vector<rectangle> detectFace(matrix<rgb_pixel> &image);

    std::vector<rectangle> detectFace(cv_image<rgb_pixel> &image);

private:
    frontal_face_detector face_detect_HOG_;
    net_type face_detect_net_;
    float detect_thresh_;
    bool use_model_;

};


#endif //FACE_RECOGNITION_FACEDETECTION_H
