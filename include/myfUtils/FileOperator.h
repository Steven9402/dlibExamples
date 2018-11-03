//
// Created by cuizhou on 18-2-4.
//

#ifndef CUIZHOUOBJDETECT_FILEOPERATOR_H
#define CUIZHOUOBJDETECT_FILEOPERATOR_H


#include <stdlib.h>
#include <dirent.h>
#include <string>
#include <vector>

namespace myf{

    /**
     * 递归读取一个文件夹下所有的文件
     * @param basePath
     * @return
     */
    std::vector<std::string> readFileList(const char *basePath);

    /**
     * 返回当前文件夹下的文件夹和文件
     * @param basePath
     * @param folders
     * @param files
     */
    void walk(const char *basePath,std::vector<std::string>& folders,std::vector<std::string>& files);
}

#endif //CUIZHOUOBJDETECT_FILEOPERATOR_H
