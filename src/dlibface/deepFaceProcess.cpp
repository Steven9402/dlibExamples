//
// Created by kyxu on 17-8-30.
//
#include <opencv2/imgproc/imgproc.hpp>
#include <Common/parameterReader.h>
#include "../../include/dlibface/deepFaceProcess.h"
#define TEST_TIME
#ifdef  TEST_TIME
#include "time.h"
#endif
//#include <QTime>
//#include <QDebug>

using namespace std;

deepFaceProcess::deepFaceProcess(std::string para_dir)
{
//    face_detect_HOG_ = get_frontal_face_detector();
    face_detector_ = FaceDetection(para_dir);
    parameterReader *para_reader = parameterReader::GetInstance(para_dir);

    string face_refine_model_path = para_reader->get_data("face_refine_model_path");
//    string face_refine_model_path = "../../data/models/shape_predictor_5_face_landmarks.dat";
    deserialize(face_refine_model_path) >> face_refine_model_;

    string face_net_path = para_reader->get_data("face_model_path").c_str();
//    string face_net_path = "../../data/models/dlib_face_recognition_resnet_model_v1.dat";
    deserialize(face_net_path) >> face_descriptor_extract_net_;

    length_thresh_ = atof(para_reader->get_data("merge_length_thresh").c_str());
    use_jitter_ = atoi(para_reader->get_data("use_jitter").c_str());
    need_merge_ = atoi(para_reader->get_data("need_merge").c_str());
}

std::vector<rectangle> deepFaceProcess::detect_face(cv::Mat &image)
{
    return face_detector_.detectFace(image);
}

std::vector<rectangle> deepFaceProcess::detect_face(cv_image<rgb_pixel> &image)
{
    return face_detector_.detectFace(image);
}

std::vector<rectangle> deepFaceProcess::detect_face(matrix<rgb_pixel> &image)
{
    return face_detector_.detectFace(image);
}

std::vector<matrix<float, 0, 1>> deepFaceProcess::extract_face_descriptor(std::vector<matrix<rgb_pixel>> &face_pitches)
{
    return face_descriptor_extract_net_(face_pitches);
}

void deepFaceProcess::extract_face_descriptors(cv::Mat &image, std::vector<rectangle> face_locations, std::vector<dlib::matrix<float, 0, 1> > &face_descriptors)
{
    face_descriptors.clear();
    cv_image<rgb_pixel> img(image);
    std::vector<matrix<rgb_pixel>> face_chips;
    for(auto rec : face_locations)
    {
        full_object_detection shape = face_refine_model_(img, rec);
        matrix<rgb_pixel> face_chip;
//        extract_image_chip(img, get_face_chip_details(shape, input_rgb_image_size), face_chip);
        extract_image_chip(img, get_face_chip_details(shape, input_rgb_image_size, 0.25), face_chip);
        face_chips.push_back(move(face_chip));
    }

    if(use_jitter_)
    {
        for(size_t i = 0; i < face_chips.size(); ++i)
        {
            matrix<float, 0, 1> face_descriptor = mean(mat(face_descriptor_extract_net_(jitter_image(face_chips[i]))));
            face_descriptors.push_back(face_descriptor);
        }
    }
    else
    {
        face_descriptors = face_descriptor_extract_net_(face_chips);
    }
//    std::vector<matrix<float, 0, 1>> face_descriptors = face_descriptor_extract_net_(face_chips);
    if(need_merge_ && face_descriptors.size() != 0)
    {
        std::vector<sample_pair> edges;

        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            for(size_t j = i + 1; j < face_descriptors.size(); ++j)
            {
                if(length(face_descriptors[i] - face_descriptors[j]) < length_thresh_)
                    edges.push_back(sample_pair(i, j));
            }
        }
        std::vector<unsigned long> labels;
        const auto num_clusters = chinese_whispers(edges, labels);

        std::vector<matrix<float, 0, 1>> face_descriptors_merged;
//        std::vector<image_window> win_clusters(num_clusters);
        for(size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            bool has_push = false;
//            std::vector<matrix<rgb_pixel>> temp;
            for(size_t j = 0; j < labels.size(); ++j)
            {
                if(cluster_id == labels[j])
                {
//                    temp.push_back(face_chips[j]);
                    if(!has_push)
                    {
                        face_descriptors_merged.push_back(face_descriptors[j]);
                        has_push = true;
                    }
                }
            }
//            win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
//            win_clusters[cluster_id].set_image(tile_images(temp));
//            cout<<"anythind"<<endl;
        }
        face_descriptors = face_descriptors_merged;
    }
}

matrix<float, 0, 1> deepFaceProcess::extract_face_descriptor(cv::Mat &image)
{
    cv::resize(image, image, cv::Size(input_rgb_image_size, input_rgb_image_size));
    cv_image<rgb_pixel> img(image);
    matrix<rgb_pixel> matrix_img;
    assign_image(matrix_img, img);
    std::vector<matrix<rgb_pixel>> face_chip;
    face_chip.push_back(matrix_img);

    std::vector<matrix<float, 0, 1>> face_descriptors = face_descriptor_extract_net_(face_chip);
//    matrix<float, 0, 1> face_descriptor = face_descriptors[0];
    return face_descriptors[0];
}

//std::vector<matrix<float, 0, 1>> deepFaceProcess::find_face_and_extract_descriptor(cv::Mat &image, bool use_jitter, bool need_merge)
void deepFaceProcess::find_face_and_extract_descriptor(cv::Mat &image, std::vector<matrix<float, 0, 1>> &face_descriptors,
                                                       std::vector<rectangle> &face_locations) //, bool use_jitter, bool need_merge
{
#ifdef  TEST_TIME
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    clock_t start_detect, finish_detect;
    double duration_detect;
    start_detect = clock();
#endif
    face_descriptors.clear();
    face_locations.clear();
    cv_image<rgb_pixel> img(image);

//    QTime test_time;
//    test_time.start();
    std::vector<rectangle> face_recs = detect_face(img);
//    qDebug() << "detect use time " << test_time.elapsed()/1000.0 << " s.";
#ifdef  TEST_TIME
    finish_detect = clock();
    duration_detect = (double)(finish_detect - start_detect) / CLOCKS_PER_SEC;
    std::cout<<"detect costs: "<<duration_detect<<"s"<<std::endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "detect cost: " << time_used.count() << " seconds." << endl;
#endif

#ifdef  TEST_TIME
    clock_t start_refine, finish_refine;
    double duration_refine;
    start_refine = clock();
#endif
    face_locations = face_recs;
    std::vector<matrix<rgb_pixel>> face_chips;
//    test_time.start();
    for(auto rec : face_recs)
    {
        full_object_detection shape = face_refine_model_(img, rec);
        matrix<rgb_pixel> face_chip;
//        extract_image_chip(img, get_face_chip_details(shape, input_rgb_image_size), face_chip);
        extract_image_chip(img, get_face_chip_details(shape, input_rgb_image_size, 0.25), face_chip);
        face_chips.push_back(move(face_chip));
    }
//    qDebug() << "refine use time " << test_time.elapsed()/1000.0 << " s.";
#ifdef  TEST_TIME
    finish_refine = clock();
    duration_refine = (double)(finish_refine - start_refine) / CLOCKS_PER_SEC;
    std::cout<<"refine costs: "<<duration_refine<<"s"<<std::endl;
#endif


#ifdef  TEST_TIME
    clock_t start_128, finish_128;
    double duration_128;
    start_128 = clock();
    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
#endif
//    std::vector<matrix<float, 0, 1>> face_descriptors;
//    test_time.start();
    if(use_jitter_)
    {
        for(size_t i = 0; i < face_chips.size(); ++i)
        {
            matrix<float, 0, 1> face_descriptor = mean(mat(face_descriptor_extract_net_(jitter_image(face_chips[i]))));
            face_descriptors.push_back(face_descriptor);
        }
    }
    else
    {
        face_descriptors = face_descriptor_extract_net_(face_chips);
    }
//    qDebug() << "extract feature use time " << test_time.elapsed()/1000.0 << " s.";
//    std::vector<matrix<float, 0, 1>> face_descriptors = face_descriptor_extract_net_(face_chips);
#ifdef  TEST_TIME
    finish_128 = clock();
    duration_128 = (double)(finish_128 - start_128) / CLOCKS_PER_SEC;
    std::cout<<"extract 128 costs:"<<duration_128<<"s"<<std::endl;
    chrono::steady_clock::time_point t6 = chrono::steady_clock::now();
    chrono::duration<double> time_used3 = chrono::duration_cast<chrono::duration<double>>(t6 - t5);
    cout << "extract 128 costsdetect cost time " << time_used3.count() << " seconds." << endl;
    std::cout<<"--------------------------------------"<<std::endl;
#endif
    if(need_merge_ && face_descriptors.size() != 0)
    {
        std::vector<sample_pair> edges;

        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            for(size_t j = i + 1; j < face_descriptors.size(); ++j)
            {
                if(length(face_descriptors[i] - face_descriptors[j]) < length_thresh_)
                    edges.push_back(sample_pair(i, j));
            }
        }
        std::vector<unsigned long> labels;
        const auto num_clusters = chinese_whispers(edges, labels);

        std::vector<matrix<float, 0, 1>> face_descriptors_merged;
//        std::vector<image_window> win_clusters(num_clusters);
        for(size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            bool has_push = false;
//            std::vector<matrix<rgb_pixel>> temp;
            for(size_t j = 0; j < labels.size(); ++j)
            {
                if(cluster_id == labels[j])
                {
//                    temp.push_back(face_chips[j]);
                    if(!has_push)
                    {
                        face_descriptors_merged.push_back(face_descriptors[j]);
                        has_push = true;
                    }
                }
            }
//            win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
//            win_clusters[cluster_id].set_image(tile_images(temp));
//            cout<<"anythind"<<endl;
        }
        face_descriptors = face_descriptors_merged;
    }

}

void deepFaceProcess::find_face_and_extract_descriptor(matrix<rgb_pixel> &img, std::vector<matrix<float, 0, 1>> &face_descriptors,
                                                       std::vector<rectangle> &face_locations) //, bool use_jitter, bool need_merge
{
    face_descriptors.clear();
    face_locations.clear();
//    cv_image<rgb_pixel> img(image);
    std::vector<rectangle> face_recs = detect_face(img);
    face_locations = face_recs;
    std::vector<matrix<rgb_pixel>> face_chips;
    for(auto rec : face_recs)
    {
        auto shape = face_refine_model_(img, rec);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, input_rgb_image_size), face_chip);
        face_chips.push_back(move(face_chip));
    }
//    std::vector<matrix<float, 0, 1>> face_descriptors;
    if(use_jitter_)
    {
        for(size_t i = 0; i < face_chips.size(); ++i)
        {
            matrix<float, 0, 1> face_descriptor = mean(mat(face_descriptor_extract_net_(jitter_image(face_chips[i]))));
            face_descriptors.push_back(face_descriptor);
        }
    }
    else
        face_descriptors = face_descriptor_extract_net_(face_chips);
//    std::vector<matrix<float, 0, 1>> face_descriptors = face_descriptor_extract_net_(face_chips);
    if(need_merge_ && face_descriptors.size() != 0)
    {
        std::vector<sample_pair> edges;

        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            for(size_t j = i + 1; j < face_descriptors.size(); ++j)
            {
                if(length(face_descriptors[i] - face_descriptors[j]) < length_thresh_)
                    edges.push_back(sample_pair(i, j));
            }
        }
        std::vector<unsigned long> labels;
        const auto num_clusters = chinese_whispers(edges, labels);

        std::vector<matrix<float, 0, 1>> face_descriptors_merged;
//        std::vector<image_window> win_clusters(num_clusters);
        for(size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            bool has_push = false;
//            std::vector<matrix<rgb_pixel>> temp;
            for(size_t j = 0; j < labels.size(); ++j)
            {
                if(cluster_id == labels[j])
                {
//                    temp.push_back(face_chips[j]);
                    if(!has_push)
                    {
                        face_descriptors_merged.push_back(face_descriptors[j]);
                        has_push = true;
                    }
                }
            }
//            win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
//            win_clusters[cluster_id].set_image(tile_images(temp));
//            cout<<"anythind"<<endl;
        }
        face_descriptors = face_descriptors_merged;
//        return face_descriptors_merged;
    }
//    return face_descriptors;
}

std::vector<matrix<rgb_pixel>> deepFaceProcess::jitter_image(const matrix<rgb_pixel> &img)
{
    thread_local random_cropper cropper;
    cropper.set_chip_dims(input_rgb_image_size,input_rgb_image_size);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_size(0.99999);
    cropper.set_background_crops_fraction(0);
//    cropper.set_min_object_size(0.97, 0.97);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);
    std::vector<matrix<rgb_pixel>> crops;

    matrix<rgb_pixel> temp;
    for (int i = 0; i < 100; ++i)
    {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}

