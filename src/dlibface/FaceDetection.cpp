//
// Created by kyxu on 17-11-6.
//

#include "../../include/dlibface/FaceDetection.h"
#include "Common/parameterReader.h"
using namespace std;

FaceDetection::FaceDetection(){

}
FaceDetection::FaceDetection(std::string parameter_path) //float detect_thresh, std::string model_path
{
    parameterReader *para_reader = parameterReader::GetInstance(parameter_path);
    detect_thresh_ = atof(para_reader->get_data("detect_face_thresh").c_str());
    string face_detector_net_path = para_reader->get_data("face_detect_model_path").c_str();
    face_detect_HOG_ = get_frontal_face_detector();
    use_model_ = atoi(para_reader->get_data("detect_face_use_model").c_str());
    deserialize(face_detector_net_path) >> face_detect_net_;
}

std::vector<rectangle> FaceDetection::detectFace(cv::Mat &image)
{
    if(use_model_)
    {
        cv_image<rgb_pixel> cv_img(image);
        matrix<rgb_pixel> img;
        assign_image(img, cv_img);

        std::vector<rectangle> result;
        std::vector<mmod_rect> dets = face_detect_net_(img);
        for(int i = 0; i < dets.size(); i++)
        {
            if(dets[i].detection_confidence > detect_thresh_)
                result.push_back(rectangle(dets[i]));
        }
        return result;
    }
    else
    {
        cv_image<rgb_pixel> img(image);
        return face_detect_HOG_(img, detect_thresh_);
    }
}

std::vector<rectangle> FaceDetection::detectFace(matrix<rgb_pixel> &image)
{
    if(use_model_)
    {

        std::vector<rectangle> result;
        std::vector<mmod_rect> dets = face_detect_net_(image);
        for(int i = 0; i < dets.size(); i++)
        {
            if(dets[i].detection_confidence > detect_thresh_)
                result.push_back(rectangle(dets[i]));
        }
        return result;
    }
    else
        return face_detect_HOG_(image, detect_thresh_);
}

std::vector<rectangle> FaceDetection::detectFace(cv_image<rgb_pixel> &image)
{
    if(use_model_)
    {
        matrix<rgb_pixel> img;
        assign_image(img, image);
        std::vector<rectangle> result;
        std::vector<mmod_rect> dets = face_detect_net_(img);
        for(int i = 0; i < dets.size(); i++)
        {
            if(dets[i].detection_confidence > detect_thresh_)
                result.push_back(rectangle(dets[i]));
        }
        return result;
    }
    else
        return face_detect_HOG_(image, detect_thresh_);
}

//void FaceDetection::setDetectModel(std::string model_path)
//{
//    deserialize(model_path) >> face_detect_net_;
//}
//
//void FaceDetection::setDetectThresh(float detect_thresh)
//{
//    detect_thresh_ = detect_thresh;
//}
