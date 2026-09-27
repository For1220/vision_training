#include<opencv2/opencv.hpp>
#include<ceres/ceres.h>
#include<iostream>
#include<cmath>

using namespace cv;

int main() {

    VideoCapture cap("resources/task_3.mp4");
    if(!cap.isOpened()) { printf("no\n");return 0; }

    double fps = cap.get(CAP_PROP_FPS);
    int totF = (int)cap.get(CAP_PROP_FRAME_COUNT);
    VideoWriter writer("result/task3_windmill/present.mp4",
                        VideoWriter::fourcc('m', 'p', '4', 'v'),
                        fps,
                        Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                        (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
        VideoWriter writer2("result/task3_windmill/HSV.mp4",
                        VideoWriter::fourcc('m', 'p', '4', 'v'),
                        fps,
                        Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                        (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    if (!writer.isOpened()) {
        printf("VideoWriter 打开失败\n");
        return 0;
    }

    int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    printf("分辨率: %d x %d\n", W, H);
    printf("帧率: %.2f fps\n", fps);
    printf("总帧数: %d\n", totF);
    printf("时长: %.2f 秒\n", totF / fps);
    // imwrite("result/task3_windmill/pt.png",plot);


    int id = 0;
    Mat plot,frame,hsv;
    Mat tmp,masktmp,ptmp;
    std::vector< std::vector<Point> >contmp;
    std::vector< Point >rTemplate;
    FileStorage fs2("resources/r_template.yml", FileStorage::READ);
    fs2["rTemplate"] >> rTemplate;
    int tar=267;
    int gap=5;
    int pos[10000][2],Cnt[10000],missCnt[10000],wt[10000][2],CntP=0;
    bool nsm[10000],ism[10000];
    bool usedPos[10000] = {false};
    bool usedWt[10000]  = {false};
    Point2f cor[10000][4];
    while(cap.read(frame)) {
        plot=frame.clone();    
        cvtColor(plot,hsv,COLOR_BGR2HSV);
        Mat maskLow,maskHigh,mask;
        inRange(hsv, Scalar(0, 150, 50), Scalar(10, 255, 255), maskLow);
        inRange(hsv, Scalar(170, 150, 50), Scalar(179, 255, 255), maskHigh);
        bitwise_or(maskLow,maskHigh,mask);

        
        // 处理二值图
        Mat kernel =getStructuringElement(MORPH_RECT,Size(3,3));
        // dilate(mask,mask,kernel);
        // erode(mask,mask,kernel);
        // morphologyEx(mask,mask,MORPH_OPEN ,kernel);
        morphologyEx(mask,mask,MORPH_CLOSE,kernel);
        //目前处理，先做开运算在膨胀一次
        // dilate(mask,mask,kernel);

        // 法1：S上限取小=240 然后闭合运算
        // morphologyEx(mask,mask,MORPH_CLOSE,kernel);
        // 法2：H上限8,V上限230

        // morphologyEx(mask,mask,MORPH_CLOSE,kernel);
        Mat maskRGB;
        cvtColor(mask, maskRGB, COLOR_GRAY2BGR);
        writer2.write(maskRGB);

        std::vector< std::vector<Point> >contours;
        // findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        std::vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        for(size_t i=0;i<contours.size();i++) {
            // if(contours.size()==1)continue;
            if (hierarchy[i][3] != -1) continue;   // 只处理外轮廓

            double area=contourArea(contours[i]);
            // 实际图形面积
            if (area < 200) continue;      // 小噪点
            
            Moments m = moments(contours[i]);
            if(m.m00==0)continue;
            double cx = m.m10 / m.m00;
            double cy = m.m01 / m.m00;
            
            //来提取R标吧！
            // if(fabs(cx-878)<30 && fabs(cy-544)<30 && id ==tar ) { 
            //     rTemplate=contours[i]; 
            //     printf("成功提取 R 标模板，点数: %zu\n", rTemplate.size());
            // }
            double score=0;
            score = matchShapes(contours[i], rTemplate, CONTOURS_MATCH_I1, 0);

            //用minAreaRect来过滤细长条带
            RotatedRect rotRect = minAreaRect(contours[i]);
            //rectArea 矩形面积
            double rectArea = rotRect.size.width * rotRect.size.height;
            if (rectArea > 10000) continue;    // 大干扰
            Point2f corners[4];
            rotRect.points(corners);

            double w = rotRect.size.width;
            double h = rotRect.size.height;
            double ratio = max(w, h) / min(w, h);   // 真实长宽比
            // printf("真实长宽比 = %.2f\n", ratio);
            if (ratio < 0.3 || ratio > 1.4) continue;   // 过滤细长条
            // 0927 debugNo1:加填充率筛选，去除空心的目标
            double fillR=area / rectArea;
            if(rectArea == 0 || fillR<0.1)continue;
                //现在我要区分标靶已经命中和未命中
            double outerArea = contourArea(contours[i]);
            double innerArea = 0;
            int childIdx = hierarchy[i][2];
            while (childIdx != -1) {
                innerArea += contourArea(contours[childIdx]);
                childIdx = hierarchy[childIdx][0];
            }
            double holeRatio = innerArea / outerArea;

                //空洞率过大的直接排除
            // if(holeRatio > 0.73)continue;
            int holeCount = 0;
            childIdx = hierarchy[i][2];   // 第一个子轮廓
            while (childIdx != -1) {
                holeCount++;
                childIdx = hierarchy[childIdx][0];   // 下一个兄弟
            }
            Mat singleContourMask = Mat::zeros(mask.size(), CV_8UC1);
            drawContours(singleContourMask, contours, (int)i, Scalar(255), -1);   // 填充轮廓
            double whiteArea = countNonZero(singleContourMask & mask);
            double realfill= whiteArea/rectArea;
            // 用真实率来判断
            if(realfill<0.25) continue;      // 过滤已经击中的标靶
            // if(0.73<realfill && realfill<0.9) continue; // 过滤科技核心
            //点与点之间连线

            //最终判断：实际面积+相似度
            if( rectArea<1000 && score>=0.120)continue;
            bool matched = false;
            for (int p = 0; p < CntP; p++) {
                if (usedPos[p]) continue;
                int dx = pos[p][0] - (int)cx;
                int dy = pos[p][1] - (int)cy;
                if (dx*dx + dy*dy <= GAP*GAP) {
                    // 匹配成功：刷新位置，命中数 +1，丢失数清零
                    pos[p][0] = (int)cx;
                    pos[p][1] = (int)cy;
                    Cnt[p]++;
                    missCnt[p] = 0;
                    for (int j = 0; j < 4; j++) cor[p][j] = corners[j];
                    usedPos[p] = true;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                for (int w = 0; w < Lwait; w++) {
                    if (usedWt[w]) continue;
                    int dx = wt[w][0] - (int)cx;
                    int dy = wt[w][1] - (int)cy;
                    if (dx*dx + dy*dy <= GAP*GAP) {
                        // 匹配成功：更新候选，命中数 +1
                        wt[w][0] = (int)cx;
                        wt[w][1] = (int)cy;
                        wtHit[w]++;
                        for (int j = 0; j < 4; j++) wtCor[w][j] = corners[j];
                        usedWt[w] = true;
                        matched = true;

                        // ★ 出现 2 帧后，提升到 pos（下一帧开始画框）
                        if (wtHit[w] >= 2 && CntP < 10000) {
                            pos[CntP][0] = (int)cx;
                            pos[CntP][1] = (int)cy;
                            Cnt[CntP] = 2;          // 已经连续命中 2 帧
                            missCnt[CntP] = 0;
                            nsm[CntP] = true;       // 立即进入画框状态
                            for (int j = 0; j < 4; j++) cor[CntP][j] = corners[j];
                            CntP++;
                        }
                        break;
                    }
                }
            }
            if (!matched && Lwait < 10000) {
                wt[Lwait][0] = (int)cx;
                wt[Lwait][1] = (int)cy;
                wtHit[Lwait] = 1;
                for (int j = 0; j < 4; j++) wtCor[Lwait][j] = corners[j];
                Lwait++;
            }
            for (int p = 0; p < CntP; p++) {
                if (!usedPos[p]) {
                    missCnt[p]++;
                    Cnt[p] = 0;      // 命中数清零
                }
            }

            // 5. 更新画框状态
            for (int p = 0; p < CntP; p++) {
                // 连续命中 2 帧 → 画框
                if (Cnt[p] >= 2) nsm[p] = true;
                // 连续丢失 2 帧 → 停止画框
                if (missCnt[p] >= 2) nsm[p] = false;
            }

            // 6. 移除长时间未命中的 pos 目标（回收内存）
            int newP = 0;
            for (int p = 0; p < CntP; p++) {
                if (missCnt[p] < 10) {          // 丢失超过 10 帧就彻底丢弃
                    if (newP != p) {
                        pos[newP][0]     = pos[p][0];
                        pos[newP][1]     = pos[p][1];
                        Cnt[newP]        = Cnt[p];
                        missCnt[newP]    = missCnt[p];
                        nsm[newP]        = nsm[p];
                        for (int j = 0; j < 4; j++) cor[newP][j] = cor[p][j];
                    }
                    newP++;
                }
            }
            CntP = newP;

            // 7. 清理等待列表：移除本帧未被匹配的候选项
            int newW = 0;
            for (int w = 0; w < Lwait; w++) {
                if (usedWt[w] && wtHit[w] < 2) {  // 保留已存在但尚未提升的候选
                    if (newW != w) {
                        wt[newW][0] = wt[w][0];
                        wt[newW][1] = wt[w][1];
                        wtHit[newW] = wtHit[w];
                        for (int j = 0; j < 4; j++) wtCor[newW][j] = wtCor[w][j];
                    }
                    newW++;
                }
            }
            Lwait = newW;

            // 8. 画框：只有 nsm[p] 为 true 的目标才画
            for (int p = 0; p < CntP; p++) {
                if (nsm[p]) {
                    for (int j = 0; j < 4; j++)
                        line(plot, cor[p][j], cor[p][(j+1)%4], Scalar(0, 0, 255), 2);
                }
            }
            // printf("轮廓 %zu: 面积=%.0f, 中心=(%.0f, %.0f)\n",
            //     i, area, cx, cy);
            if(id==tar) {
            printf("轮廓 %zu: 面积=%.0f,实际矩形面积=%.0f 中心=(%.0f, %.0f) 填充率=%.3f 空洞率 %.3f 白色像素个数 %.2f 真实率 %.2f 相似度 %.3f\n",
                i, area, rectArea, cx, cy ,fillR,holeRatio,whiteArea,realfill,score);
            }
        }    
        char frameText[64];
        sprintf(frameText, "Frame: %d", id);
        putText(plot, frameText, Point(30, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
        writer.write(plot);  
        if(id==tar) {
            tmp=frame.clone();
            masktmp=mask.clone();
            ptmp=plot.clone();  
            for(size_t i=0;i<contours.size();i++) 
                contmp.push_back(contours[i]);
        }
        id++;
    }

    // cvtColor(ptmp,hsv,COLOR_BGR2HSV);
    // struct Sample { int x, y; const char* name; };
    // Sample samples[] = {
    //     {860, 513, "R标"},
    //     {836, 601, "传送带"},   // ← 你确认后的坐标
    //     {999,979, "科技核心"},
    //     {1000,957,"科技核心错识别"},
    //     {731,703,"靶形目标下方区域"},
    //     {717,703,"标靶不可识别区域估计"}
    // };
    // // 该步骤用于已知目标坐标求其HSV参数，推荐软件 GIMP
    // for (auto& s : samples) {
    //     Vec3b p = hsv.at<Vec3b>(s.y, s.x);
    //     printf("%s (%d, %d) -> H=%d, S=%d, V=%d\n",
    //         s.name, s.x, s.y, p[0], p[1], p[2]);
    // }

    // //以下片段用于单帧的check
    //     for(size_t i=0;i<contmp.size();i++) {
    //         double area=contourArea(contmp[i]);
    //         if (area < 200) continue;      // 小噪点
            
    //         Moments m = moments(contmp[i]);
    //         if(m.m00==0)continue;
    //         double cx = m.m10 / m.m00;
    //         double cy = m.m01 / m.m00;
            
    //         //用minAreaRect来过滤细长条带
    //         RotatedRect rotRect = minAreaRect(contmp[i]);
    //         double rectArea = rotRect.size.width * rotRect.size.height;
    //         if (rectArea > 10000) continue;    // 大干扰
    //         Point2f corners[4];
    //         rotRect.points(corners);

    //         double w = rotRect.size.width;
    //         double h = rotRect.size.height;
    //         double ratio = max(w, h) / min(w, h);   // 真实长宽比

    //         if (ratio < 0.3 || ratio > 1.4) continue;   // 过滤细长条
    //         // 0927 debugNo1:加填充率筛选，去除空心的目标
    //         double fillR=area / rectArea;
    //         if(rectArea == 0 || fillR<0.1)continue;
    //         // printf("轮廓 %zu: 面积=%.0f,实际矩形面积=%.0f 中心=(%.0f, %.0f) 填充率=%.3f \n",
    //         //     i, area, rectArea, cx, cy ,fillR);

    //     }    

    imwrite("result/task3_windmill/red_mask.png",masktmp);
    imwrite("result/task3_windmill/check.png",tmp);
    imwrite("result/task3_windmill/plot.png",ptmp);

    // FileStorage fs("r_template.yml", FileStorage::WRITE);
    // fs << "rTemplate" << rTemplate;
    // fs.release();

    fs2.release();
    writer.release();
    cap.release();
    writer2.release();
    return 0;
}