#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlibface/faceDescriptorManager.h>
#include "myfUtils/MatOperation.h"

using namespace cv;
using namespace std;

int main()
{
    char* parameter_dir = "../../res/parameters.ini";
    char* save_dir = "../../data/facepool";
    faceDescriptorManager face_descriptor_manager(parameter_dir);

    VideoCapture capture(0);

    double width=capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double height=capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    width = 1920;
    height = 1080;
    width = 1280;
    height = 720;
    capture.set(CV_CAP_PROP_FRAME_WIDTH,width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,height);

    Mat frame;
    int count=0;
    int frameidx=0;
    while(capture.read(frame))
    {
        if(frame.empty())
            continue;

        if(frameidx%5==0) {

            std::vector<dlib::rectangle> facerects = face_descriptor_manager.detect_face(frame);

            for (dlib::rectangle facerect:facerects) {
                double left = facerect.left();
                auto top = facerect.top();
                auto right = facerect.right();
                auto bottom = facerect.bottom();


                Rect expandedROI;
                myf::expandRoi(100, Rect(Point(left, top), Point(right, bottom)), expandedROI, frame.cols, frame.rows);

                Mat facemat;
                frame(expandedROI).copyTo(facemat);

                stringstream oo;
                oo << save_dir << "/" << count << ".jpg" << endl;
                imwrite(oo.str(), facemat);

                cv::rectangle(frame,Point(left,top),Point(right,bottom),Scalar(0,255,255),1,8,0);
                stringstream ooo;
                ooo<<count;
                cv::putText(frame,ooo.str(),Point(left,top),FONT_HERSHEY_COMPLEX,1.0,Scalar(255,255,0),1,8,0);
                count++;
            }
        }

        frameidx++;
        cv::imshow("result", frame);
        cv::waitKey(33);
    }
    capture.release();
    return 1;
}