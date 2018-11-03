//
// Created by cuizhou on 18-2-25.
//

#ifndef MSERPROPOSEREGION_RECTANGLEOPERATION_H
#define MSERPROPOSEREGION_RECTANGLEOPERATION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
namespace myf{

    class RectangleOperation {

    public:
        RectangleOperation(int);

        double calcIouRatio(const Rect &rect1, const Rect &rect2);

        //重叠部分面积除以小的矩形面积,大于0.8可以认为小矩形被遮盖。
        double calcOverlapRatio(const Rect &rect1, const Rect &rect2);
    };
}



#endif //MSERPROPOSEREGION_RECTANGLEOPERATION_H
