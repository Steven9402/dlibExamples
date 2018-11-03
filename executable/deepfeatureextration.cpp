//
// Created by cuizhou on 18-3-19.
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlibface/deepFaceProcess.h>

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
    char* parameter_path = "../../res/parameters.ini";

    Mat srcImage = imread(imgdir);

    deepFaceProcess deepface_impl(parameter_path);

//    std::vector<dlib::rectangle> facerects = deepface_impl.detect_face(srcImage);


    //cv::Mat small_frame;
    std::vector<dlib::rectangle> face_locations;
    std::vector<dlib::matrix<float, 0, 1>> face_encodings;
    deepface_impl.find_face_and_extract_descriptor(srcImage, face_encodings, face_locations);

    for(dlib::matrix<float, 0, 1> face_descriptor:face_encodings){
        cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
    }


    for(dlib::rectangle facerect:face_locations){
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
