#include<opencv2/opencv.hpp>
#include<iostream>
#include<cmath>
#include<vector>

using namespace cv;

// ★ 手动设置调试帧号
int tar = 526;

struct Track {
    int x, y;
    int hitCount;
    int missCount;
    bool locked;
    Point2f corners[4];
    double area;
    double score;
    double realfill;
    double rectArea;
    int id;
};

struct IdRecord {
    int id;
    int x, y;
    int lastSeenFrame;
};

int main() {
    VideoCapture cap("resources/task_4.mp4");// task_3 or task_4
    if(!cap.isOpened()) { printf("no\n"); return 0; }

    double fps = cap.get(CAP_PROP_FPS);
    int totF = (int)cap.get(CAP_PROP_FRAME_COUNT);
    VideoWriter writer("result/task3_windmill/task_4/recognition_overlay.mp4",
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       fps,
                       Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                            (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    VideoWriter writer2("result/task3_windmill/HSV.mp4",
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       fps,
                       Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                            (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    if (!writer.isOpened()) { printf("VideoWriter 打开失败\n"); return 0; }

    int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    printf("分辨率: %d x %d\n", W, H);
    printf("调试帧号 tar = %d\n", tar);

    int id = 0;
    Mat plot, frame, hsv;
    std::vector<Point> rTemplate;
    FileStorage fs2("resources/r_template.yml", FileStorage::READ);
    if (fs2.isOpened()) { fs2["rTemplate"] >> rTemplate; fs2.release(); }
    printf("模板点数: %zu\n", rTemplate.size());

    const int LOCK_FRAMES     = 3;
    const int MISS_LIMIT      = 2;
    const int TRACK_REMOVE    = 9;
    const double R_RECT_MAX   = 1500.0;
    const double R_SCORE_MAX  = 0.13;

    const int R_LOCK_FRAMES   = 3;
    const int R_MISS_LIMIT    = 5;

    const int GAP_BIG = 80;
    const int R_TRACK_GAP   = 60;
    const int R_LOST_RESET  = 30;

    const int ID_LOST_LIMIT = 10;

    Track tracks[10000];
    int trackCount = 0;

    IdRecord idRecords[100];
    int idRecordCount = 0;
    int bladeIdCounter = 0;

    int rX = -1, rY = -1;
    Point2f rCorners[4];
    int rHitCount = 0;
    int rMissCount = 0;
    bool rLocked = false;
    bool rEverLocked = false;

    while(cap.read(frame)) {
        plot = frame.clone();
        cvtColor(plot, hsv, COLOR_BGR2HSV);
        Mat maskLow, maskHigh, mask;
        inRange(hsv, Scalar(0, 150, 50), Scalar(10, 255, 255), maskLow);
        inRange(hsv, Scalar(170, 150, 50), Scalar(179, 255, 255), maskHigh);
        bitwise_or(maskLow, maskHigh, mask);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        Mat maskRGB;
        cvtColor(mask, maskRGB, COLOR_GRAY2BGR);
        writer2.write(maskRGB);

        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        // ===== 收集候选 =====
        struct Cand {
            double cx, cy, area, score, rectArea, realfill;
            Point2f corners[4];
            bool isSmall;
            bool isRPot;
        };
        std::vector<Cand> candidates;

        for (size_t i = 0; i < contours.size(); i++) {
            if (hierarchy[i][3] != -1) continue;
            double area = contourArea(contours[i]);
            if (area < 200) continue;

            Moments m = moments(contours[i]);
            if (m.m00 == 0) continue;
            double cx = m.m10 / m.m00, cy = m.m01 / m.m00;

            double score = 0;
            if (!rTemplate.empty()) {
                score = matchShapes(contours[i], rTemplate, CONTOURS_MATCH_I1, 0);
            }

            RotatedRect rotRect = minAreaRect(contours[i]);
            double rectArea = rotRect.size.width * rotRect.size.height;
            if (rectArea > 10000) continue;
            Point2f corners[4];
            rotRect.points(corners);

            double w = rotRect.size.width, h = rotRect.size.height;
            double ratio = max(w, h) / min(w, h);
            if (ratio < 0.3 || ratio > 1.4) continue;

            double fillR = area / rectArea;
            if (rectArea == 0 || fillR < 0.1) continue;

            Mat singleMask = Mat::zeros(mask.size(), CV_8UC1);
            drawContours(singleMask, contours, (int)i, Scalar(255), -1);
            double whiteArea = countNonZero(singleMask & mask);
            double realfill = whiteArea / rectArea;

            if (realfill < 0.25) continue;

            bool isSmall = (rectArea < R_RECT_MAX);
            if (isSmall && score >= R_SCORE_MAX) continue;

            Cand c;
            c.cx = cx; c.cy = cy;
            c.area = area; c.score = score;
            c.rectArea = rectArea; c.realfill = realfill;
            for (int j = 0; j < 4; j++) c.corners[j] = corners[j];
            c.isSmall = isSmall;
            c.isRPot = (isSmall && score < R_SCORE_MAX);
            candidates.push_back(c);
        }

        // ===== ★ 调试输出：id==tar 时输出所有被框选的目标 =====
        if (id == tar) {
            printf("\n===== Frame %d 所有被框选目标 =====\n", id);
            printf("候选总数: %zu\n", candidates.size());
            for (size_t i = 0; i < candidates.size(); i++) {
                auto& c = candidates[i];
                printf("[候选 %zu] 中心=(%.1f, %.1f)  面积=%.1f  rectArea=%.1f  真实占有率=%.3f  相似度score=%.4f\n",
                       i, c.cx, c.cy,
                       c.area, c.rectArea,
                       c.realfill, c.score);
            }
            printf("====================================\n\n");
        }

        // ===== R 标独立跟踪 =====
        std::vector<int> rCandIndices;
        for (size_t i = 0; i < candidates.size(); i++) {
            if (candidates[i].isRPot) rCandIndices.push_back((int)i);
        }

        int rFinalIdx = -1;
        if (!rCandIndices.empty()) {
            if (rX < 0 || !rLocked) {
                double bestScore = 1e9;
                for (int idx : rCandIndices) {
                    if (candidates[idx].score < bestScore) {
                        bestScore = candidates[idx].score;
                        rFinalIdx = idx;
                    }
                }
            } else {
                double bestDist = 1e9;
                int nearestIdx = -1;
                for (int idx : rCandIndices) {
                    double dx = candidates[idx].cx - rX;
                    double dy = candidates[idx].cy - rY;
                    double d = sqrt(dx*dx + dy*dy);
                    if (d < bestDist) {
                        bestDist = d;
                        nearestIdx = idx;
                    }
                }
                if (nearestIdx >= 0 && bestDist <= R_TRACK_GAP) {
                    rFinalIdx = nearestIdx;
                } else if (rMissCount >= R_LOST_RESET) {
                    double bestScore = 1e9;
                    for (int idx : rCandIndices) {
                        if (candidates[idx].score < bestScore) {
                            bestScore = candidates[idx].score;
                            rFinalIdx = idx;
                        }
                    }
                }
            }
        }

        if (rFinalIdx >= 0) {
            rX = (int)candidates[rFinalIdx].cx;
            rY = (int)candidates[rFinalIdx].cy;
            for (int j = 0; j < 4; j++) rCorners[j] = candidates[rFinalIdx].corners[j];
            rHitCount++;
            rMissCount = 0;
            if (rHitCount >= R_LOCK_FRAMES) {
                rLocked = true;
                rEverLocked = true;
            }
            candidates[rFinalIdx].isRPot = false;
        } else {
            rMissCount++;
            rHitCount = 0;
            if (rMissCount >= R_MISS_LIMIT) rLocked = false;
        }

        // ===== 靶子 track =====
        bool usedTrack[10000] = {false};
        for (auto& c : candidates) {
            if (c.isSmall) continue;
            if (rLocked && fabs(c.cx - rX) < 3 && fabs(c.cy - rY) < 3) continue;

            bool matched = false;
            for (int t = 0; t < trackCount; t++) {
                if (usedTrack[t]) continue;
                int dx = tracks[t].x - (int)c.cx;
                int dy = tracks[t].y - (int)c.cy;
                if (dx*dx + dy*dy <= GAP_BIG*GAP_BIG) {
                    tracks[t].x = (int)c.cx;
                    tracks[t].y = (int)c.cy;
                    tracks[t].hitCount++;
                    tracks[t].missCount = 0;
                    tracks[t].area = c.area;
                    tracks[t].score = c.score;
                    tracks[t].realfill = c.realfill;
                    tracks[t].rectArea = c.rectArea;
                    for (int j = 0; j < 4; j++) tracks[t].corners[j] = c.corners[j];
                    if (tracks[t].hitCount >= LOCK_FRAMES) tracks[t].locked = true;
                    usedTrack[t] = true;
                    matched = true;
                    break;
                }
            }

            if (!matched && trackCount < 10000) {
                tracks[trackCount].x = (int)c.cx;
                tracks[trackCount].y = (int)c.cy;
                tracks[trackCount].hitCount = 1;
                tracks[trackCount].missCount = 0;
                tracks[trackCount].locked = false;
                tracks[trackCount].area = c.area;
                tracks[trackCount].score = c.score;
                tracks[trackCount].realfill = c.realfill;
                tracks[trackCount].rectArea = c.rectArea;
                tracks[trackCount].id = 0;
                for (int j = 0; j < 4; j++) tracks[trackCount].corners[j] = c.corners[j];
                usedTrack[trackCount] = true;
                trackCount++;
            }
        }

        for (int t = 0; t < trackCount; t++) {
            if (!usedTrack[t]) {
                tracks[t].missCount++;
                tracks[t].hitCount = 0;
            }
        }
        int newCount = 0;
        for (int t = 0; t < trackCount; t++) {
            if (tracks[t].missCount < TRACK_REMOVE) {
                if (newCount != t) tracks[newCount] = tracks[t];
                newCount++;
            }
        }
        trackCount = newCount;

        // ===== 靶子 ID 分配 =====
        for (int t = 0; t < trackCount; t++) {
            if (!tracks[t].locked || tracks[t].id != 0) continue;
            if (tracks[t].missCount > 0) continue;

            int matchedId = 0;
            for (int k = 0; k < idRecordCount; k++) {
                if (idRecords[k].lastSeenFrame < id - 60) continue;
                int dx = idRecords[k].x - tracks[t].x;
                int dy = idRecords[k].y - tracks[t].y;
                if (dx*dx + dy*dy <= 100*100) {
                    matchedId = idRecords[k].id;
                    idRecords[k].x = tracks[t].x;
                    idRecords[k].y = tracks[t].y;
                    idRecords[k].lastSeenFrame = id;
                    break;
                }
            }

            if (matchedId != 0) {
                tracks[t].id = matchedId;
            } else {
                bladeIdCounter++;
                tracks[t].id = bladeIdCounter;
                printf("[Frame %d] 新分配 Blade%d, pos=(%d,%d)\n",
                       id, bladeIdCounter, tracks[t].x, tracks[t].y);
                idRecords[idRecordCount].id = tracks[t].id;
                idRecords[idRecordCount].x = tracks[t].x;
                idRecords[idRecordCount].y = tracks[t].y;
                idRecords[idRecordCount].lastSeenFrame = id;
                idRecordCount++;
            }
        }

        for (int t = 0; t < trackCount; t++) {
            if (tracks[t].id == 0) continue;
            for (int k = 0; k < idRecordCount; k++) {
                if (idRecords[k].id == tracks[t].id) {
                    idRecords[k].x = tracks[t].x;
                    idRecords[k].y = tracks[t].y;
                    idRecords[k].lastSeenFrame = id;
                    break;
                }
            }
        }

        // ===== 画 R 标 =====
        if (rLocked) {
            for (int j = 0; j < 4; j++) {
                line(plot, rCorners[j], rCorners[(j+1)%4], Scalar(255, 0, 0), 2);
            }
            circle(plot, Point(rX, rY), 6, Scalar(255, 0, 0), -1);
            putText(plot, "R", Point(rX - 10, rY - 30),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 100, 0), 2);
        }

        // ===== 画靶子 =====
        for (int t = 0; t < trackCount; t++) {
            if (tracks[t].id <= 0) continue;
            if (!tracks[t].locked) continue;
            bool detected = (tracks[t].missCount < MISS_LIMIT);
            if (!detected) continue;

            for (int j = 0; j < 4; j++) {
                line(plot, tracks[t].corners[j], tracks[t].corners[(j+1)%4],
                     Scalar(0, 0, 255), 2);
            }

            char idOnlyText[32];
            sprintf(idOnlyText, "Blade%d", tracks[t].id);
            putText(plot, idOnlyText,
                    Point(tracks[t].x - 40, tracks[t].y - 30),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

            if (rLocked) {
                circle(plot, Point(tracks[t].x, tracks[t].y), 6, Scalar(0, 255, 0), -1);
                line(plot, Point(rX, rY), Point(tracks[t].x, tracks[t].y),
                     Scalar(0, 255, 255), 2);
            }
        }

        // ===== 左上角显示 =====
        int textY = 70;
        const int LINE_HEIGHT = 30;

        if (rEverLocked && rMissCount <= ID_LOST_LIMIT) {
            bool rDetected = (rMissCount < MISS_LIMIT && rLocked);
            char idText[64];
            sprintf(idText, "R: %s", rDetected ? "detected" : "lost");
            Scalar color = rDetected ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(plot, idText, Point(30, textY),
                    FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            textY += LINE_HEIGHT;
        }

        for (int t = 0; t < trackCount; t++) {
            if (tracks[t].id <= 0) continue;
            if (tracks[t].missCount > ID_LOST_LIMIT) continue;

            bool tDetected = (tracks[t].missCount < MISS_LIMIT);
            char idText[128];

            if (tDetected && rLocked) {
                double dx = tracks[t].x - rX;
                double dy = rY - tracks[t].y;
                double theta = atan2(dy, dx);
                sprintf(idText, "Blade%d: detected  angle=%.2f rad",
                        tracks[t].id, theta);
            } else {
                sprintf(idText, "Blade%d: %s", tracks[t].id,
                        tDetected ? "detected" : "lost");
            }

            Scalar color = tDetected ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(plot, idText, Point(30, textY),
                    FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            textY += LINE_HEIGHT;
        }

        char frameText[64];
        sprintf(frameText, "Frame: %d", id);
        putText(plot, frameText, Point(30, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
        writer.write(plot);
        id++;
    }

    writer.release();
    cap.release();
    writer2.release();
    return 0;
}