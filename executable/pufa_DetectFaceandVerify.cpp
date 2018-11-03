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
    float score_threshold_ = 0.5;

    faceDescriptorManager face_descriptor_manager(parameter_dir);


    VideoCapture capture("../../data/test.mp4");

    double width=capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double height=capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    width = 1920;
    height = 1080;
    width = 1280;
    height = 720;
    capture.set(CV_CAP_PROP_FRAME_WIDTH,width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,height);

    Mat frame;
    while(capture.read(frame)) {
        if (frame.empty())
            continue;

        std::vector<matrix<float, 0, 1>> face_encodings;
        std::vector<dlib::rectangle> face_locations;
        face_descriptor_manager.find_face_and_extract_descriptor(frame, face_encodings, face_locations);


        for (int i = 0; i < face_encodings.size(); i++) {

            std::vector<std::pair<std::string, float> > simi_name_scores;
            std::pair<std::string, float> simi_name_score;

            face_descriptor_manager.recognize_face(face_encodings[i], 3, simi_name_scores);
            simi_name_score = simi_name_scores[0];

            string name;
            if (simi_name_score.second > score_threshold_) {
                name = "unknown";
            } else {
                name = simi_name_score.first;
            }

            double left = face_locations[i].left();
            double top = face_locations[i].top();
            double right = face_locations[i].right();
            double bottom = face_locations[i].bottom();

            cv::rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 255), 1, 8, 0);
            cv::putText(frame, name, Point(left, top), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 0), 1, 8, 0);
        }

        cv::imshow("result", frame);
        cv::waitKey(33);
    }
    capture.release();
    return 1;
}
