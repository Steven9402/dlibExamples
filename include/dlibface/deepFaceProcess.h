//
// Created by kyxu on 17-8-30.
//

#ifndef DEEPFACEPROCESS_H
#define DEEPFACEPROCESS_H

//namespace CZ_FACE{


#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <dlib/opencv.h>
#include <vector>

#include "FaceDetection.h"

using namespace dlib;

const int input_rgb_image_size = 150;//atoi(pd->get_data("input_face_image_size").c_str());

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, the jittering you can see below in jitter_image() was used during
// training, and the training dataset consisted of about 3 million images instead of 55.
// Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                             alevel0<
                                             alevel1<
                                             alevel2<
                                             alevel3<
                                             alevel4<
                                             max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                             input_rgb_image_sized<input_rgb_image_size>
                                             >>>>>>>>>>>>;

/*/ ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------*/

class deepFaceProcess {
public:
    deepFaceProcess(std::string para_dir);
    ~deepFaceProcess(){}

public:
    std::vector<rectangle> detect_face(cv::Mat &image);

    std::vector<rectangle> detect_face(cv_image<rgb_pixel> &image);

    std::vector<rectangle> detect_face(matrix<rgb_pixel> &image);

    std::vector<matrix<float, 0, 1>> extract_face_descriptor(std::vector<matrix<rgb_pixel>> &face_pitches);

    void extract_face_descriptors(cv::Mat &image, std::vector<dlib::rectangle> face_locations, std::vector<dlib::matrix<float, 0, 1>> &face_descriptors);

    void find_face_and_extract_descriptor(cv::Mat &image, std::vector<matrix<float, 0, 1>> &face_descriptors,
                                          std::vector<rectangle> &face_locations);  //, bool use_jitter = false, bool need_merge = false

    void find_face_and_extract_descriptor(matrix<rgb_pixel> &img, std::vector<matrix<float, 0, 1>> &face_descriptors,
                                          std::vector<rectangle> &face_locations); //, bool use_jitter = false, bool need_merge = false

    matrix<float, 0, 1> extract_face_descriptor(cv::Mat &image);


private:
    std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel> &img);

private:
    FaceDetection face_detector_;
//    frontal_face_detector face_detect_HOG_;
    shape_predictor face_refine_model_;
    anet_type face_descriptor_extract_net_;

//    net_type face_detect_net_;
//    float detect_thresh_;
//    bool detect_use_model_;

private:
    float length_thresh_;
    bool use_jitter_;
    bool need_merge_;
};
//}

#endif //FACE_RECOGNITION_DEEPFACEPROCESS_H
