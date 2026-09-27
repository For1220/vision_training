#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;

int main() {
    Mat img = imread("resources/test_image.jpg");
    if( img.empty() ) {
        printf("no\n");
        return 0;
    }
    resize(img , img , Size(), 0.3, 0.3);  // 宽高都缩小到 30%
    // imshow("Original",img); 
    // 1.1 

    Mat gray;
    cvtColor(img , gray ,COLOR_BGR2GRAY);
    imwrite("result/task1_images/gray.png",gray); 
    // 1.2

    Mat channels[3],hsv;
    cvtColor(img , hsv ,COLOR_BGR2HSV);
    split(hsv,channels);
    imwrite("result/task1_images/hsv_h.png",channels[0]);
    imwrite("result/task1_images/hsv_s.png",channels[1]);
    imwrite("result/task1_images/hsv_v.png",channels[2]);

    Mat meanImg,gaussianImg,medianImg;
    blur(img , meanImg , Size(5,5) );
    GaussianBlur(img , gaussianImg, Size(5,5), 1,5);
    medianBlur(img , medianImg , 5);
    imwrite("result/task1_images/mean_filter.png",meanImg);
    imwrite("result/task1_images/gaussian_filter.png",gaussianImg);
    imwrite("result/task1_images/median_filter.png",medianImg);

    Mat maskLow,maskHigh,mask;
    cvtColor(img,hsv,COLOR_BGR2HSV);
    // 源代码
    // inRange(hsv,Scalar(0,130,80),Scalar(10,255,255),maskLow);
    // inRange(hsv,Scalar(170,130,80),Scalar(179,255,255),maskHigh);
    // ds 优化
    inRange(hsv, Scalar(0, 110, 60), Scalar(10, 255, 255), maskLow);
    inRange(hsv, Scalar(170, 110, 60), Scalar(179, 255, 255), maskHigh);
    bitwise_or(maskLow,maskHigh,mask);
    imwrite("result/task1_images/red_mask.png",mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat dilated, eroded, opened, closed;
    dilate(mask , dilated, kernel);
    erode(mask , eroded, kernel);
    morphologyEx(mask , opened, MORPH_OPEN, kernel);
    morphologyEx(mask , closed, MORPH_CLOSE, kernel);    
    imwrite("result/task1_images/dilate.png",dilated);
    imwrite("result/task1_images/erode.png",eroded);
    imwrite("result/task1_images/open.png",opened);
    imwrite("result/task1_images/close.png",closed);

    Mat morph,edges,pre;
    morphologyEx( mask , pre , MORPH_OPEN , kernel);
    morphologyEx(pre , pre , MORPH_CLOSE , kernel);
    Canny(pre, edges, 100, 200);
    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, morph, MORPH_CLOSE, kernel);
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    // findContours(morph, contours, hierarchy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // 试一试直接对 close 进行提取 
    // 9/23 22:11 果然直接对 close 提取效果好，ds 解释：Canny会让边缘变成双线，使得面积变小
    findContours(closed , contours, hierarchy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contourImg = img.clone();
    for(int i=0;i < contours.size(); i++) {
        double area= contourArea( contours[i] );
        if(area < 600) continue;
        drawContours(contourImg, contours , i , Scalar(0,255,0) ,2);
        // printf("%d %f\n",i,area);
        Rect box = boundingRect( contours[i] );
        rectangle(contourImg , box , Scalar(0,0,255) , 2 );
    }
    imwrite("result/task1_images/contours_boxes.png",contourImg);

    Mat drawing = img.clone();
    circle(drawing, Point(300, 200), 80, Scalar(0, 255, 0), 3);                   
    rectangle(drawing, Point(500, 100), Point(750, 350), Scalar(0, 0, 255), 3);
    putText(drawing,"Hello,RM!",Point(100, 500),FONT_HERSHEY_SIMPLEX,1.5,Scalar(255,0,0),3); 
    imwrite("result/task1_images/drawing.png",drawing);

    Point2f center(img.cols / 2.0, img.rows / 2.0);
    Mat rotmat = getRotationMatrix2D(center,35,1.0),rMat;
    warpAffine(img,rMat,rotmat,img.size());
    imwrite("result/task1_images/rotated_35deg.png",rMat);

    Rect roi(0,0,img.cols / 2.0, img.rows / 2.0);
    Mat cropped = img(roi);
    imwrite("result/task1_images/crop_top_left.png",cropped);
    waitKey(0);
    return 0;
}