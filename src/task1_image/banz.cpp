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
    // imshow("Original",img); 1.1

    Mat gray;
    cvtColor(img , gray ,COLOR_BGR2GRAY);
    // imshow("Gray",gray); 1.2

    Mat meanImg,gaussianImg,medianImg;
    blur(img , meanImg , Size(5,5) );
    GaussianBlur(img , gaussianImg, Size(5,5), 1,5);
    medianBlur(img , medianImg , 5);
    // imshow("Mean", img );
    // imshow("Gaussian", gaussianImg);
    // imshow("Median",medianImg); 1.3

    Mat binary, adaptive;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 128, 255, THRESH_BINARY);
    adaptiveThreshold(gray, adaptive, 255,
    ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    // imshow("Binary", binary);
    // imshow("Adaptive", adaptive); 1.4

    Mat smooth,edges;
    GaussianBlur(gray , smooth ,Size(5,5) ,1.5);
    Canny(smooth,edges,100,200);
    // imshow("Canny",edges); 1.5

    Mat hsv,maskLow,maskHigh,mask;
    cvtColor(img,hsv,COLOR_BGR2HSV);
    inRange(hsv,Scalar(0,100,100),Scalar(10,255,255),maskLow);
    inRange(hsv,Scalar(170,100,100),Scalar(179,255,255),maskHigh);
    bitwise_or(maskLow,maskHigh,mask);
    // imshow("Red mask",mask); 1.6

    Mat equalized;
    cvtColor(img,gray,COLOR_BGR2GRAY);
    equalizeHist(gray,equalized);
    // imshow("Equealized",equalized); 1.7

    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 128, 255, THRESH_BINARY);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dilated, eroded, opened, closed;
    dilate(binary, dilated, kernel);
    erode(binary, eroded, kernel);
    morphologyEx(binary, opened, MORPH_OPEN, kernel);
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);
    // imshow("Open", opened);
    // imshow("Close", closed); 1.8

    Mat gradX, gradY, absX, absY, grad;
    Sobel(gray, gradX, CV_16S, 1, 0);
    Sobel(gray, gradY, CV_16S, 0, 1);
    convertScaleAbs(gradX, absX);
    convertScaleAbs(gradY, absY);
    addWeighted(absX, 0.5, absY, 0.5, 0, grad);
    // imshow("Sobel", grad); 1.9 Sobel

    Mat channels[3],colorEdges;
    cvtColor(img,hsv,COLOR_BGR2HSV);
    split(hsv,channels);
    Canny(channels[0],colorEdges,100,200);
    // imshow("Hue edges",colorEdges); 1.10 颜色边缘检测

    Mat morph;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 100, 200);
    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, morph, MORPH_CLOSE, kernel);
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(morph, contours, hierarchy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contourImg = Mat::zeros(img.size(), CV_8UC3);
    drawContours(contourImg, contours, -1, Scalar(0, 255, 0), 2);
    // imshow("Contours", contourImg); 1.11 绘制轮廓

    Mat result = img.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < 500.0) continue;
        Rect box = boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;
        rectangle(result, box, Scalar(0, 0, 255), 2);
        drawContours(result, contours, static_cast<int>(i),
        Scalar(0, 255, 0), 2);
    }
    // imshow("Filtered contours", result); 1.12 用外接矩形框选目标
    // 已知问题：轮廓绘制尚可，但矩形框选有点问题，可能是筛选参数未调整好

    
    waitKey(0);
    return 0;
}