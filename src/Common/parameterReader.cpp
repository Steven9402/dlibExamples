//
// Created by kyxu on 17-9-2.
//

#include "../../include/Common/parameterReader.h"
using namespace std;

parameterReader::parameterReader(std::string filename)
{
    data_.clear();
    ifstream fin(filename.c_str());
    if(!fin)
    {
        cerr << "Can not open parameter file " << filename<<endl;
        return;
    }
    while(!fin.eof())
    {
        string str;
        getline(fin, str);
        if(str[0] == '#')
            continue;

        int pos = str.find("=");
        if(pos == -1)
            continue;
        string key = str.substr(0, pos);
        string value = str.substr(pos + 1, str.length());
        data_[key] = value;
        if(!fin.good())
            break;
    }
    fin.close();
}

parameterReader::~parameterReader()
{
    if(instance_ != NULL)
        delete instance_;
}

string parameterReader::get_data(std::string key)
{
    map<string, string>::const_iterator iter = data_.find(key);
    if(iter == data_.end())
    {
        cerr << "Parameter name " << key << " not found!" << endl;
        return string("NOT_FOUND");
    }
    return iter->second;
}

parameterReader *parameterReader::instance_ = NULL;
parameterReader *parameterReader::GetInstance(std::string init_file_path)
{
    if(instance_ == NULL)
        instance_ = new parameterReader(init_file_path);
    return instance_;
//    if(instance_ == NULL)
//        instance_ = new parameterReader(init_file_path);
//    else
//    {
////        delete instance_;
//        instance_ = NULL;
//        instance_ = new parameterReader(init_file_path);
//    }
//    return instance_;
}