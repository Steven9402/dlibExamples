//
// Created by kyxu on 17-8-31.
//

#ifndef FACEDESCRIPTORMANAGER_H
#define FACEDESCRIPTORMANAGER_H

#include <memory>
#include <map>
#include <string>
#include "deepFaceProcess.h"
#include "protoFaceEncode.pb.h"
#include "myfUtils/FileOperator.h"

class faceDescriptorManager {

public:
    faceDescriptorManager(std::string para_dir);
    ~faceDescriptorManager();

    /**
     * give a dir contains all known people, init the face encoding lib
     */
    void init_face_encoding(std::string image_dir);
    void init_face_encoding_multi(std::string image_dir);


    /**
     * add a face encode to the face encode lib
     */
    void add_face(std::string name, dlib::matrix<float, 0, 1> face_encode);

    /**
     * give a face encode, recognize who she/he is from the face encode lib
     */
    void recognize_face(const dlib::matrix<float, 0, 1> &face_encode, std::pair<std::string, float> &name_score);

    /**
     * give a face encode, return top num score and name
     */
    void recognize_face(const dlib::matrix<float, 0, 1> &face_encode, int top_num, std::vector<std::pair<std::string, float> > &name_scores);

    /**
     * @brief detect_face
     * @param image
     * @return
     */
    std::vector<rectangle> detect_face(cv::Mat &image);
//
//    std::vector<rectangle> detect_face(matrix<rgb_pixel> &image);

    void find_face_and_extract_descriptor(cv::Mat &image, std::vector<matrix<float, 0, 1>> &face_descriptors, std::vector<rectangle> &face_locations);

    /**
     * @brief extract_face_descriptor
     * @param face_locations
     * @param face_descriptors
     */
    void extract_face_descriptors(cv::Mat &image, std::vector<dlib::rectangle> face_locations, std::vector<dlib::matrix<float, 0, 1>> &face_descriptors);


private:
    /**
     *
     */
    static bool compire(const std::pair<std::string, float> a, const std::pair<std::string ,float> b);

    void saveFacePatch(std::string img_path, dlib::rectangle face_rect, std::string face_path_dir);

    /**
    * give a dir, get all the file with specify suffix
    */
    std::vector<std::string> get_all_files(std::string path, std::string suffix = ".*\.jpg");
    std::map<std::string, std::string> get_all_specify_files(std::string path, std::string suffix = ".*\.jpg");

    /*
     * get face encodes form a pb file when the system start
     */
    void from_proto();
    void from_txt();

    /**
     * transform a proto face encode to matrix face encode
     */
    matrix<float, 0, 1> trans_proto_matrix(const protoFaceEncode &proto_face_encode);


    /*
     * save all face encodes to a pb file when the system end
     */
    void to_proto();
    void to_txt();

    /**
     * transform a matrix face encode to proto face encode
     */
    protoFaceEncode trans_matrix_ptoro(const matrix<float, 0, 1> &face_encode);

private:
    std::shared_ptr<deepFaceProcess> deep_face_processor_;
    std::map<std::string, dlib::matrix<float, 0, 1>> face_encodings_;
    std::map<std::string, dlib::matrix<float, 0, 1>> face_encodings_multi_;

    float show_face_border_ratio_;

    //门禁系统人员样本库
    std::string face_patch_save_dir_;
    std::string face_encode_file_path_;

    std::string face_encode_file_txt_path_;
    bool use_proto_;

    bool use_multi_verification_;
};




#endif //FACE_RECOGNITION_FACEDESCRIPTORMANAGER_H
