//
// Created by cuizhou on 18-3-19.
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlibface/faceDescriptorManager.h>
#include "myfUtils/FileOperator.h"
//#define FAMILIAR_ONLY


using namespace cv;
using namespace std;
using namespace dlib;
#ifdef FAMILIAR_ONLY

int main()
{
    /**
     * 在准备好的 熟人（train test），陌生人库上测试识别准确率
     * step0:  ../data/facepool 放置人脸库
     * step1: 设置 parameters.ini init_face_lib=1
     * step2: 设置 parameter_path
     * step3: 设置训练集
     * step4: 设置测试集
     */


    char* parameter_dir = "../../res/parameters.ini";
    float score_threshold_ = 0.5;

    faceDescriptorManager face_descriptor_manager(parameter_dir);

    std::vector<string> input_root_path_test_lists = {"/media/NEWDATA/data/sfz-org/库-真人-stay"};//step2: 设置测试集

    //test
    for(int ind=0;ind<input_root_path_test_lists.size();ind++) {
//        ofstream f1(output_txt_name[ind]);
        string input_root_path_test = input_root_path_test_lists[ind];//step2: 设置测试集
        std::vector<string> folders_test;
        std::vector<string> files_test;
        myf::walk(input_root_path_test.c_str(), folders_test, files_test);

        int recog_correct_count=0;
        int recog_success_count=0;
        int recog_failed_count=0;

        for (string folder_test:folders_test) {
            string sub_input_root = input_root_path_test;
            sub_input_root += "/" + folder_test;
            std::vector<string> filenames = myf::readFileList(sub_input_root.c_str());

            for (string filename:filenames) {
                //读图
                string image_input_path = sub_input_root + "/" + filename;//图片输入路径
                Mat srcImage = cv::imread(image_input_path);

                //识别
                float score;


                std::vector<matrix<float, 0, 1>> face_encodings;
                std::vector<dlib::rectangle> face_locations;
                face_descriptor_manager.find_face_and_extract_descriptor(srcImage, face_encodings, face_locations);

                int maxid=0;
                int maxheight=0;
                for(int ind=0;ind<face_locations.size();ind++){
                    int height=face_locations[ind].bottom()-face_locations[ind].top();
                    if(height>maxheight){
                        maxheight=height;
                        maxid=ind;
                    }
                }
                if(face_encodings.size()==0)continue;



                std::vector<std::pair<std::string, float> > simi_name_scores;
                std::pair<std::string, float> simi_name_score;

                face_descriptor_manager.recognize_face(face_encodings[maxid], 3, simi_name_scores);
                simi_name_score = simi_name_scores[0];

                //门禁模式
                string  groundtruth = folder_test;
                string recognize_result = simi_name_score.first;

                cout<<"groundtruth = "<<groundtruth<<"  result = "<<recognize_result<<endl;

                string name;
                if (simi_name_score.second > score_threshold_) {
                    recog_failed_count++;
                } else {
                    if (recognize_result == groundtruth) {
                        recog_correct_count++;
                    }
                    recog_success_count++;
                }


            }
        }

        cout<<"test result for: "<<input_root_path_test_lists[ind]<<endl;
        cout<<"recog_correct_count = "<<recog_correct_count<<endl;
        cout<<"recog_success_count = "<<recog_success_count<<endl;
        cout<<"accuracy            = "<<(float)recog_correct_count/recog_success_count<<endl;
        cout<<"recog_failed_count  = "<<recog_failed_count<<endl;

    }

    return 1;
}
#else


int main()
{
    /**
     * 在准备好的 熟人（train test），陌生人库上测试识别准确率
     * step0:  ../data/facepool 放置人脸库
     * step1: 设置 parameters.ini init_face_lib=1
     * step2: 设置 parameter_path
     * step3: 设置训练集
     * step4: 设置测试集
     */


    char* parameter_dir = "../../res/parameters.ini";
    float score_threshold_ = 0.5;

    faceDescriptorManager face_descriptor_manager(parameter_dir);

    std::vector<string> input_root_path_test_lists = {"/home/cuizhou/AAAA/BoardFACE/data/original/chinese_celebrity_face_pool/whole/test",
                                                 "/home/cuizhou/AAAA/BoardFACE/data/original/chinese_celebrity_face_pool/whole/stranger"};//step2: 设置测试集

    //test
    for(int ind=0;ind<input_root_path_test_lists.size();ind++) {
//        ofstream f1(output_txt_name[ind]);
        string input_root_path_test = input_root_path_test_lists[ind];//step2: 设置测试集
        std::vector<string> folders_test;
        std::vector<string> files_test;
        myf::walk(input_root_path_test.c_str(), folders_test, files_test);

        int recog_correct_count=0;
        int recog_success_count=0;
        int recog_failed_count=0;

        for (string folder_test:folders_test) {
            string sub_input_root = input_root_path_test;
            sub_input_root += "/" + folder_test;
            std::vector<string> filenames = myf::readFileList(sub_input_root.c_str());

            for (string filename:filenames) {
                //读图
                string image_input_path = sub_input_root + "/" + filename;//图片输入路径
                Mat srcImage = cv::imread(image_input_path);

                //识别
                float score;


                std::vector<matrix<float, 0, 1>> face_encodings;
                std::vector<dlib::rectangle> face_locations;
                face_descriptor_manager.find_face_and_extract_descriptor(srcImage, face_encodings, face_locations);

                int maxid=0;
                int maxheight=0;
                for(int ind=0;ind<face_locations.size();ind++){
                    int height=face_locations[ind].bottom()-face_locations[ind].top();
                    if(height>maxheight){
                        maxheight=height;
                        maxid=ind;
                    }
                }
                if(face_encodings.size()==0)continue;


                std::vector<std::pair<std::string, float> > simi_name_scores;
                std::pair<std::string, float> simi_name_score;

                face_descriptor_manager.recognize_face(face_encodings[maxid], 3, simi_name_scores);
                simi_name_score = simi_name_scores[0];

                //门禁模式
                string  groundtruth = folder_test;
                string recognize_result = simi_name_score.first;

                cout<<"groundtruth = "<<groundtruth<<"  result = "<<recognize_result<<endl;

                string name;
                if (simi_name_score.second > score_threshold_) {
                    recog_failed_count++;
                } else {
                    if (recognize_result == groundtruth) {
                        recog_correct_count++;
                    }
                    recog_success_count++;
                }


            }
        }

        cout<<"test result for: "<<input_root_path_test_lists[ind]<<endl;
        cout<<"recog_correct_count = "<<recog_correct_count<<endl;
        cout<<"recog_success_count = "<<recog_success_count<<endl;
        cout<<"accuracy            = "<<(float)recog_correct_count/recog_success_count<<endl;
        cout<<"recog_failed_count  = "<<recog_failed_count<<endl;

        ofstream f1("../../data/result.txt",std::ios::out|std::ios::app);
        f1<<"test result for: "<<input_root_path_test_lists[ind]<<endl;
        f1<<"recog_correct_count = "<<recog_correct_count<<endl;
        f1<<"recog_success_count = "<<recog_success_count<<endl;
        f1<<"accuracy            = "<<(float)recog_correct_count/recog_success_count<<endl;
        f1<<"recog_failed_count  = "<<recog_failed_count<<endl;
        f1.close();

    }

    return 1;
}
#endif
