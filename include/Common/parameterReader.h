//
// Created by kyxu on 17-9-2.
//

#ifndef PARAMETERREADER_H
#define PARAMETERREADER_H

#include <string>
#include <iostream>
#include <map>
#include <fstream>

class parameterReader {
private:
    parameterReader(std::string filename);

public:
    static parameterReader *GetInstance(std::string init_file_path = "../../res/parameters.ini");
    std::string get_data(std::string key);
    ~parameterReader();

private:
    std::map<std::string, std::string> data_;
    static parameterReader *instance_;
};


#endif //FACE_RECOGNITION_PARAMETERREADER_H
