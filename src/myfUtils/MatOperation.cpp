//
// Created by cuizhou on 18-2-24.
//

#include "myfUtils/MatOperation.h"
namespace  myf {

    cv::Mat composeGrayVisMat(std::vector<Mat> matpool) {
        int cols = 20;
        int rows = matpool.size() / 20 + 1;

        int size = 75;

        Mat vismat = Mat::zeros(size * rows, size * cols, CV_8UC1);
        for (int ind = 0; ind < matpool.size(); ind++) {
            int row = ind / cols;
            int col = ind % cols;
            int x = col * size;
            int y = row * size;
            Rect pasteroi(x, y, size - 5, size - 5);

            Mat pastemat;
            resize(matpool[ind], pastemat, Size(size - 5, size - 5), 0, 0);
            pastemat.copyTo(vismat(pasteroi));
        }

        return vismat;
    }

    bool expandRoi(int expandwidth, const cv::Rect &roi, cv::Rect &expandedROI, int matwidth, int matheight) {
        if (matwidth < 1 || matheight < 1)return false;

        int x = roi.x - expandwidth;
        int y = roi.y - expandwidth;
        int x2 = roi.br().x + expandwidth;
        int y2 = roi.br().y + expandwidth;

        if (x < 0)x = 0;
        if (y < 0)y = 0;
        if (x2 > matwidth - 1)x2 = matwidth - 1;
        if (y2 > matheight - 1)y2 = matheight - 1;
        expandedROI = Rect(Point(x, y), Point(x2, y2));

        return true;
    }
}