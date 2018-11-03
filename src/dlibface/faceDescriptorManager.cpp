//
// Created by kyxu on 17-8-31.
//
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dirent.h>
#include <fstream>
#include <regex>
#include <chrono>

#include "dlibface/faceDescriptorManager.h"
#include <Common/parameterReader.h>

using namespace std;
using namespace dlib;

faceDescriptorManager::faceDescriptorManager(std::string para_dir)
{

    parameterReader *para_reader = parameterReader::GetInstance(para_dir);

    deep_face_processor_ = std::make_shared<deepFaceProcess>(para_dir);
    face_encodings_.clear();
    show_face_border_ratio_ = atof(para_reader->get_data("show_face_border_ratio").c_str());
    face_encode_file_path_ = para_reader->get_data("face_encode_file").c_str();
    face_patch_save_dir_ = para_reader->get_data("face_patch_path");
    face_encode_file_txt_path_ = para_reader->get_data("face_encode_file_txt").c_str();

    use_proto_ = atoi(para_reader->get_data("use_proto").c_str());

    use_multi_verification_ =  atoi(para_reader->get_data("use_multi_verification").c_str());

    bool init_face_lib = atoi(para_reader->get_data("init_face_lib").c_str());
    if(init_face_lib)
    {
        string image_path = para_reader->get_data("face_lib_image_path").c_str();

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        if(use_multi_verification_)init_face_encoding_multi(image_path);
        else init_face_encoding(image_path);

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "Init face lib from images cost time " << time_used.count() << " secondes." << endl;
    }
    else
    {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        if(use_proto_)from_proto();
        else from_txt();

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "Read proto file cost time " << time_used.count() << " secondes." << endl;
    }

}
faceDescriptorManager::~faceDescriptorManager()
{
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    if(use_proto_)to_proto();
    else to_txt();

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Save proto file cost time " << time_used.count() << " secondes." << endl;
}

void faceDescriptorManager::init_face_encoding_multi(std::string image_dir){

    std::vector<std::string> folders;
    std::vector<std::string> files;
    const char* basedir = image_dir.c_str();
    myf::walk(basedir,folders,files);

    std::map<std::string,std::vector<string>> person_files;

    for(std::string personname:folders){
        std::string person_base_dir=image_dir+"/"+personname;

        std::vector<std::string> personpics = myf::readFileList(person_base_dir.c_str());
        person_files[personname]=personpics;
    }

    std::map<std::string, std::vector<std::string>>::iterator iter;
    for(iter = person_files.begin(); iter != person_files.end(); iter++){
        std::vector<std::string> personpics = iter->second;
        for(std::string pathname:personpics){
            std::string imgpath = image_dir+"/"+iter->first+"/"+pathname;
            cout<<imgpath<<endl;
            matrix<rgb_pixel> img;
            load_image(img, imgpath);
            std::vector<matrix<float, 0, 1>> face_encodes;
            std::vector<rectangle> face_locations;
            deep_face_processor_->find_face_and_extract_descriptor(img, face_encodes, face_locations);
            if(face_encodes.size()==0)continue;
//            face_encodings_multi_[iter->first] = face_encodes[0];
            face_encodings_[iter->first] = face_encodes[0];
        }
    }
}



void faceDescriptorManager::init_face_encoding(std::string image_dir)
{
    std::map<string, string> person_files = get_all_specify_files(image_dir);
    std::map<string, string>::iterator iter;
    for(iter = person_files.begin(); iter != person_files.end(); iter++)
    {
        cout << iter->first << " " << iter->second << endl;
        matrix<rgb_pixel> img;
        load_image(img, iter->second);
        std::vector<matrix<float, 0, 1>> face_encodes;
        std::vector<rectangle> face_locations;
        deep_face_processor_->find_face_and_extract_descriptor(img, face_encodes, face_locations);
        if(face_encodes.size()==0)continue;
        face_encodings_[iter->first] = face_encodes[0];

//        cv::Rect rect(face_locations[0].left(), face_locations[0].top(), face_locations[0].width(), face_locations[0].height());
        saveFacePatch(iter->second, face_locations[0], face_patch_save_dir_);
    }
//    cout << face_encodings_.size() << endl;
//    std::map<string, matrix<float, 0, 1>>::iterator iter_map;
//    for(iter_map = face_encodings_.begin(); iter_map != face_encodings_.end(); iter_map++)
//    {
//        cout << iter_map->first << endl << iter_map->second << endl;
//    }
    return;
}

void faceDescriptorManager::add_face(std::string name, matrix<float, 0, 1> face_encode)
{
    face_encodings_[name] = face_encode;
    return;
}

void faceDescriptorManager::recognize_face(const matrix<float, 0, 1> &face_encode, std::pair<std::string, float> &name_score)
{
    std::vector<float> scores;
    std::vector<std::string> face_names;

    std::map<string, matrix<float, 0, 1>>::const_iterator iter;

    for(iter = face_encodings_.begin(); iter != face_encodings_.end(); iter++)
    {
        auto score = length(iter->second - face_encode);
        scores.push_back(score);
        face_names.push_back(iter->first);

//        cout << score << endl;
//        if(score < min_score)
//        {
//            name = iter->first;
//            min_score = score;
//        }
    }
    std::vector<float>::iterator min_iter = std::min_element(std::begin(scores), std::end(scores));
    name_score.second = *min_iter;
    int index = std::distance(std::begin(scores), min_iter);
    name_score.first = face_names[index];

//    qDebug() << "min_score: " << min_score << " index: " << index << " name: " << QString::fromStdString(name);
    return;
}

void faceDescriptorManager::recognize_face(const dlib::matrix<float, 0, 1> &face_encode, int top_num, std::vector<std::pair<std::string, float> > &name_scores)
{
    name_scores.clear();

    std::vector<std::pair<std::string, float> > face_scores;

    std::map<string, matrix<float, 0, 1>>::const_iterator iter;
    for(iter = face_encodings_.begin(); iter != face_encodings_.end(); iter++)
    {
        auto score = length(iter->second - face_encode);
        face_scores.push_back(std::make_pair(iter->first, score));
    }
    std::sort(face_scores.begin(), face_scores.end(), compire);
    name_scores.assign(face_scores.begin(), face_scores.begin() + top_num);
}

std::vector<rectangle> faceDescriptorManager::detect_face(cv::Mat &image)
{
    return deep_face_processor_->detect_face(image);
}

void faceDescriptorManager::find_face_and_extract_descriptor(cv::Mat &image, std::vector<matrix<float, 0, 1>> &face_descriptors, std::vector<rectangle> &face_locations)
{
    deep_face_processor_->find_face_and_extract_descriptor(image, face_descriptors, face_locations);
}

void faceDescriptorManager::extract_face_descriptors(cv::Mat &image, std::vector<rectangle> face_locations, std::vector<dlib::matrix<float, 0, 1> > &face_descriptors)
{
    deep_face_processor_->extract_face_descriptors(image, face_locations, face_descriptors);
}


bool faceDescriptorManager::compire(const std::pair<string, float> a, const std::pair<string, float> b)
{
    return a.second < b.second;
}

void faceDescriptorManager::saveFacePatch(string img_path, dlib::rectangle face_rect, string face_path_dir)
{
    int first_index = img_path.find_last_of('/');
    int last_index = img_path.find_last_of('.');
    string name = img_path.substr(first_index, last_index - first_index);
    cv::Mat img = cv::imread(img_path);
    cv::Rect rect(face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height());

    std::cout << "name:" << img_path << ", ratio: " << show_face_border_ratio_;
    int border = rect.height * show_face_border_ratio_;
    rect.x -= border;
    rect.y -= border;
    rect.width += border * 2;
    rect.height += border * 2;


    if(rect.x < 0)
        rect.x = 0;
    if(rect.y < 0)
        rect.y = 0;
    if(rect.x + rect.width > img.cols)
        rect.width = img.cols - rect.x - 1;
    if(rect.y + rect.height > img.rows)
        rect.height = img.rows - rect.y - 1;

    if(rect.width < rect.height)
    {
        int sub = rect.height - rect.width;
        rect.y = rect.y + sub/2;
        rect.height = rect.height - sub;
    }else if(rect.height < rect.width)
    {
        int sub = rect.width - rect.height;
        rect.x = rect.x - sub/2;
        rect.width = rect.width - sub;
    }
    std::cout << "img: rows = " << img.rows << ", cols = " << img.cols;
    std::cout << "rect: rows = " << rect.height << ", cols = " << rect.width;

    cv::Mat face_patch;// = img(rect).clone();
    img(rect).copyTo(face_patch);
    cv::resize(face_patch, face_patch, cv::Size(150, 150));
    cv::imwrite(face_path_dir + name + ".jpg", face_patch);
}

std::vector<std::string> faceDescriptorManager::get_all_files(std::string path, std::string suffix)
{
    std::vector<std::string> files;
    files.clear();
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(path.c_str())) == NULL)
    {
        cout << "Can not open " << path << endl;
        return files;
    }
    regex reg_obj(suffix, regex::icase);
    while((dirp = readdir(dp)) != NULL)
    {
        if(dirp -> d_type == 8)  // 4 means catalog; 8 means file; 0 means unknown
        {
            if(regex_match(dirp->d_name, reg_obj))
            {
//                cout << dirp->d_name << endl;
                string full_path = path + dirp->d_name;

                files.push_back(full_path);

//                cout << dirp->d_name << " " << dirp->d_ino << " " << dirp->d_off << " " << dirp->d_reclen << " " << dirp->d_type << endl;
            }
        }
    }
    closedir(dp);
    return files;
}

std::map<std::string, std::string> faceDescriptorManager::get_all_specify_files(std::string path, std::string suffix)
{
    std::map<std::string, std::string> files;
    files.clear();
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(path.c_str())) == NULL)
    {
        cout << "Can not open " << path << endl;
        return files;
    }
    regex reg_obj(suffix, regex::icase);
    while((dirp = readdir(dp)) != NULL)
    {
        if(dirp -> d_type == 8)  // 4 means catalog; 8 means file; 0 means unknown
        {
            if(regex_match(dirp->d_name, reg_obj))
            {
//                cout << dirp->d_name << endl;
                string file_name = dirp->d_name;
                string full_path = path + file_name;
                string person_name = file_name.substr(0, file_name.length() - 4);
                files[person_name] = full_path;

//                cout << dirp->d_name << " " << dirp->d_ino << " " << dirp->d_off << " " << dirp->d_reclen << " " << dirp->d_type << endl;
            }
        }
    }
    closedir(dp);
    return files;
}


void faceDescriptorManager::to_proto()
{
    fstream output(face_encode_file_path_, ios::out | ios::trunc | ios::binary);
    protoFaceEncodes proto_face_encodes;
//    google::protobuf::Map face_encode_map = face_encodes.face_encodes();
    google::protobuf::Map<string, protoFaceEncode> *face_encode_map = proto_face_encodes.mutable_proto_face_encodes();

    std::map<string, matrix<float, 0, 1>>::const_iterator iter;
    for(iter = face_encodings_.begin(); iter != face_encodings_.end(); iter++)
    {
        string name = iter->first;
        protoFaceEncode proto_face_encode = trans_matrix_ptoro(iter->second);
        (*face_encode_map)[name] = proto_face_encode;
    }

    if(!proto_face_encodes.SerializeToOstream(&output))
    {
        cerr << "Failed to write face encode proto file." << endl;
        return;
    }
    return;
}

void faceDescriptorManager::to_txt()
{
    fstream output(face_encode_file_txt_path_, ios::out | ios::trunc);
    output << face_encodings_.size()<<endl;
    std::map<string, matrix<float, 0, 1>>::const_iterator iter;
    for(iter = face_encodings_.begin(); iter != face_encodings_.end(); iter++)
    {
        output << left << setw(16) << iter->first;
        for(int i = 0; i < iter->second.size(); ++i)
        {
            output << left << setw(16) << iter->second(i);
        }
        output << endl;
    }
    output.close();
}

void faceDescriptorManager::from_proto()
{
    fstream input(face_encode_file_path_, ios::in | ios::binary);
    if(!input)
    {
        cout << "There is no face_encodes.pb file." << endl;
        return;
    }

    protoFaceEncodes proto_face_encodes;
    if(!proto_face_encodes.ParseFromIstream(&input))
    {
        cerr << "Failed to parse face_encode file." << endl;
        return;
    }

    google::protobuf::Map<string, protoFaceEncode> proto_face_encode = proto_face_encodes.proto_face_encodes();
    google::protobuf::Map<string, protoFaceEncode>::const_iterator proto_iter;
    for(proto_iter = proto_face_encode.begin(); proto_iter != proto_face_encode.end(); proto_iter++)
    {
        string name = proto_iter->first;
        matrix<float, 0, 1> face_encode = trans_proto_matrix(proto_iter->second);
        face_encodings_[name] = face_encode;
    }
    return;
}

void faceDescriptorManager::from_txt()
{
    fstream input(face_encode_file_txt_path_, ios::in);
    if(!input)
    {
        cout << "There is no face_encodes.txt file." << endl;
        return;
    }
    int size;
    input >> size;
    cout << "Face encodes size is " << size << endl;
    string name;
    matrix<float, 0, 1> face_encode(128);
    face_encodings_.clear();
    for(int j = 0; j < size; ++j)
    {
        input >> name;
        for(int i = 0; i < 128; ++i)
        {
            input >> face_encode(i);
        }
        face_encodings_[name] = face_encode;
    }
    input.close();
}

matrix<float, 0, 1> faceDescriptorManager::trans_proto_matrix(const protoFaceEncode &proto_face_encode)
{
    matrix<float, 0, 1> face_encode(proto_face_encode.encode_size());
    for(int i = 0; i < proto_face_encode.encode_size(); ++i)
    {
        face_encode(i) = proto_face_encode.encode(i);
    }
    return face_encode;
}

protoFaceEncode faceDescriptorManager::trans_matrix_ptoro(const matrix<float, 0, 1> &face_encode)
{
    protoFaceEncode proto_face_encode;
    for(int i = 0; i < face_encode.size(); ++i)
    {
        proto_face_encode.add_encode(face_encode(i));
    }
    return proto_face_encode;
}
