//
// Created by cuizhou on 18-2-4.
//

#include <cstring>
#include <iostream>

#include "myfUtils/FileOperator.h"
namespace  myf{
    std::vector<std::string> readFileList(const char *basePath)
    {
        std::vector<std::string> result;
        DIR *dir;
        struct dirent *ptr;
        char base[1000];

        if ((dir=opendir(basePath)) == NULL)
        {
            perror("Open dir error...");
            exit(1);
        }

        while ((ptr=readdir(dir)) != NULL)
        {
            if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
                continue;
            else if(ptr->d_type == 8)    ///file
            {printf("d_name:%s/%s\n",basePath,ptr->d_name);
                result.push_back(std::string(ptr->d_name));}
            else if(ptr->d_type == 10)    ///link file
            {printf("d_name:%s/%s\n",basePath,ptr->d_name);
                result.push_back(std::string(ptr->d_name));}
            else if(ptr->d_type == 4)    ///dir
            {
                memset(base,'\0',sizeof(base));
                strcpy(base,basePath);
                strcat(base,"/");
                strcat(base,ptr->d_name);
                result.push_back(std::string(ptr->d_name));
                readFileList(base);
            }
        }
        closedir(dir);
        return result;
    }

    void walk(const char *basePath, std::vector<std::string>& folders, std::vector<std::string>& files)
    {
        DIR *dir;
        struct dirent *ptr;

        if ((dir=opendir(basePath)) == NULL)
        {
            perror("Open dir error...");
            exit(1);
        }

        while ((ptr=readdir(dir)) != NULL)
        {
            if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
                continue;
            else if(ptr->d_type == 8)    ///file
            {printf("d_name:%s/%s\n",basePath,ptr->d_name);
                files.push_back(std::string(ptr->d_name));}
            else if(ptr->d_type == 10)    ///link file
            {printf("d_name:%s/%s\n",basePath,ptr->d_name);
                files.push_back(std::string(ptr->d_name));}
            else if(ptr->d_type == 4)    ///dir
            {
                folders.push_back(std::string(ptr->d_name));
            }
        }
        closedir(dir);
    }

}
