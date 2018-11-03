// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image.  In
    particular, this program shows how you can take a list of images from the
    command line and display each on the screen with red boxes overlaid on each
    human face.

    The examples/faces folder contains some jpg images of people.  You can run
    this program on them and see the detections by executing the following command:
        ./face_detection_ex faces/*.jpg

    
    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  This type of object detector is fairly
    general and capable of detecting many types of semi-rigid objects in
    addition to human faces.  Therefore, if you are interested in making your
    own object detectors then read the fhog_object_detector_ex.cpp example
    program.  It shows how to use the machine learning tools which were used to
    create dlib's face detector. 


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace dlib;
using namespace std;
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {

        //string savepath=argv[4];

//        string inputimagefn="/home/cuizhou/myGitRepositories/dlibExamples/data/detection/2007_007763.jpg";
        string inputimagefn="/home/cuizhou/myGitRepositories/dlibExamples/data/detection/2009_004587.jpg";
        cv::Mat src = cv::imread(inputimagefn);

        frontal_face_detector detector = get_frontal_face_detector();

        cout << "processing image " << argv[1] << endl;
        array2d<unsigned char> img;
        load_image(img, inputimagefn);
        //pyramid_up(img);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces it can find in the image.
        std::vector<rectangle> dets = detector(img);

        for(int ind=0;ind<dets.size();ind++){
            char temp[4];
            sprintf(temp,"%d",ind);
            string ind_face=temp;

            cv::Mat faceMat;
            int left=dets[ind].left();
            int top=dets[ind].top();
            int right=dets[ind].right();
            int bottom=dets[ind].bottom();
            cv::Rect facerect(cv::Point(left,top),cv::Point(right,bottom));
            cout<<"=== "<<src.rows<<" "<<src.cols<<"      "<<facerect<<endl;
            src(facerect).copyTo(faceMat);
            //string outputpath=savepath+"/"+ind_image+"_"+ind_face+".jpg";
            //cv::imwrite(outputpath,faceMat);
            cv::imshow("faceMat",faceMat);
            cv::waitKey(0);
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }

    return 1;
}

// ----------------------------------------------------------------------------------------

