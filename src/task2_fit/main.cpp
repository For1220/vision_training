#include<opencv2/opencv.hpp>
#include<ceres/ceres.h>
#include<iostream>
#include<cmath>
using namespace cv;

struct OmegaCostFunctor {
    OmegaCostFunctor(double t,double o_now) : t_(t),o_now_(o_now){}

    template<typename T>
    bool operator()(const T* const params,T* residual) const {
        T A =params[0];
        T b =params[1];
        T Omega=params[2];
        T phi=params[3];

        T omega_pred = b+A*sin(Omega * T(t_) + phi);
        residual[0]=omega_pred - T(o_now_);
        return true;
    }
    double t_;
    double o_now_;
};

int main() {
    VideoCapture cap("resources/task_2.mp4");
    if(!cap.isOpened()) {
        printf("no\n");
        return 0;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int totF = (int)cap.get(CAP_PROP_FRAME_COUNT);
    
    VideoWriter writer("result/task2_fit/tracking_overlay.mp4",
                    VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps,
                    Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                            (int)cap.get(CAP_PROP_FRAME_HEIGHT)));

    if (!writer.isOpened()) {
        printf("VideoWriter 打开失败\n");
        return 0;
    }

    double cx = 480.0, cy = 360.0;
    std::vector<Point2f>centers;
    std::vector<double> times;
    std::vector<double> theta_row,theT;
    int id=0;
    Mat frame;
    while(cap.read(frame)) {
        Mat hsv,mask;
        cvtColor(frame,hsv,COLOR_BGR2HSV);
        inRange(hsv,Scalar(80,150,150),Scalar(100,255,255),mask);

        std::vector< std::vector<Point> >contours;
        findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        int tmp=0;
        double mA=0;
        Point2f center(-1,-1);
        for(int i=0;i<contours.size();i++) {
            double area = contourArea(contours[i]);
            if(area > mA && area > 50) {
                mA=area;
                tmp = i;
                Moments m= moments(contours[i]);
                center = Point2f(m.m10/m.m00,m.m01/m.m00);
            }
        }
        if (center.x > 0 && center.y > 0) {
            //计算角度
            double dx = center.x - cx;
            double dy = cy - center.y;
            double theta = atan2(dy, dx);
            char text[64];
            sprintf(text, "theta = %.2f rad", theta);
            putText(frame, text, Point(30, 50),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
            theta_row.push_back(theta);
        } 
        else {
        //没检测到目标，标明 LOST
        putText(frame, "LOST", Point(30, 50),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }
        Rect box = boundingRect( contours[tmp] );
        rectangle(frame , box, Scalar(0,0,255) , 2 );
        centers.push_back(center);
        times.push_back(id/fps);
        id++;
        writer.write(frame);
    }
    // printf("%zu %zu %zu\n",centers.size(),times.size(),theta_row.size());
    theT.push_back(theta_row[0]);
    double offset=0.0;
    for(int i=1;i<theta_row.size();i++) {
        double diff = theta_row[i]-theta_row[i-1];
        if(diff>M_PI) offset -= 2*M_PI;
        else if(diff<-M_PI) offset +=2*M_PI;
        theT.push_back(theta_row[i]+offset);
    }
    std::vector<double>omega;
    for(int i=1;i<theT.size();i++) {
        double dt=times[i]-times[i-1];
        double dth=theT[i]-theT[i-1];
        omega.push_back(dth/dt);
    }

    double omega_min = *min_element(omega.begin(), omega.end());
    double omega_max = *max_element(omega.begin(), omega.end());
    double omega_sum = 0;
    for (double w : omega) omega_sum += w;
    double omega_mean = omega_sum / omega.size();

    // printf("omega: min=%.4f, max=%.4f, mean=%.4f, range=%.4f\n",
    //     omega_min, omega_max, omega_mean, omega_max - omega_min);
    
    //FFT 操作
    Mat omegaMat(omega.size(), 1, CV_32F);
    for (size_t i = 0; i < omega.size(); i++) {
        omegaMat.at<float>(i, 0) = (float)omega[i];
    }

    // 做 DFT
    Mat planes[] = {Mat_<float>(omegaMat), Mat::zeros(omegaMat.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);

    // 计算幅值
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag = planes[0];

    // 找最大峰值（跳过 DC 分量）
    int N = omega.size();
    double maxMag = 0;
    int peakIdx = 0;
    for (int i = 1; i < N/2; i++) {
        if (mag.at<float>(i) > maxMag) {
            maxMag = mag.at<float>(i);
            peakIdx = i;
        }
    }

    // 峰值对应的频率
    double fs = fps;   // 采样频率 = 60 Hz
    double freq = (double)peakIdx * fs / N;   // 单位 Hz
    double Omega_est = 2 * M_PI * freq;       // Ω = 2πf

    // printf("主频率 = %.4f Hz, Ω_est = %.4f\n", freq, Omega_est);    
    //已完成FFT 

    // 求解problem
    double params[4] = {(omega_max - omega_min) / 2.0, omega_mean, Omega_est, 0.0};
    ceres::Problem prob;
    for(int i=0;i<omega.size();i++) {
        ceres::CostFunction* cost=
            new ceres::AutoDiffCostFunction< OmegaCostFunctor,1,4 >(
                new OmegaCostFunctor(times[i],omega[i]) );
        prob.AddResidualBlock(cost,nullptr,params);
    }

    prob.SetParameterLowerBound(params,0,0.0);
    prob.SetParameterLowerBound(params,1,0.0);
    prob.SetParameterLowerBound(params,2,0.0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres:: DENSE_QR;
    options.minimizer_progress_to_stdout=1;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&prob,&summary);

    // printf("%s\n", summary.BriefReport().c_str());
    printf("A = %.4f, b = %.4f, Ω = %.4f, φ = %.4f\n",
       params[0], params[1], params[2], params[3]);
    std::vector<double> omega_fit;
    std::vector<double> residual;
    double sse=0.0;
    for(int i=0;i<omega.size();i++) {
        double t=times[i];
        omega_fit.push_back(params[1]+params[0]*sin(params[2]*t+params[3]));
        residual.push_back(omega[i]-omega_fit[i]);
        sse += residual[i]*residual[i];
    }
    double rmse = sqrt(sse/omega.size());
    printf("RMSE = %.4f rad/s \n",rmse);
    
    int W=1200,H=600;
    Mat plot(H,W,CV_8UC3,Scalar(255,255,255));
    double t_min=times[0],t_max=times[omega.size()-1];
    double y_min=*min_element(omega.begin(),omega.end());
    double y_max=*max_element(omega.begin(),omega.end());
    auto toX = [&](double t) { return (int)((t - t_min) / (t_max - t_min) * (W - 100) + 50); };
    auto toY = [&](double y) { return (int)(H - 50 - (y - y_min) / (y_max - y_min) * (H - 100)); };
 

    for(int i=0;i<W-100;i++) {
            double t = t_min + (double)i / (W - 100) * (t_max - t_min);
        double y = params[1] + params[0] * sin(params[2] * t + params[3]);
        plot.at<Vec3b>(toY(y), toX(t)) = Vec3b(0, 0, 255);
    }   
    imwrite("result/task2_fit/angular_velocity.png",plot);
    for(int i=0;i<omega.size();i+=3) 
        circle(plot,Point(toX(times[i]),toY(omega[i])),2,Scalar(255,0,0),-1);
    char rmseText[64];
    sprintf(rmseText,"RMSE = %.4f rad/s",rmse);
    putText(plot,rmseText,Point(50,40),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,0,0),2);
    imwrite("result/task2_fit/fit_comparison.png",plot);

    Mat resPlot(H,W,CV_8UC3,Scalar(255,255,255));
    double r_min=*min_element(residual.begin(),residual.end());
    double r_max=*max_element(residual.begin(),residual.end());
    double r_abs=max(abs(r_min),abs(r_max));
    auto toResY = [&](double r) { return (int)(H/2 - r / r_abs * (H/2 - 50)); };
    for(int i=0;i<residual.size();i+=3) 
        circle(resPlot,Point(toX(times[i]),toResY(residual[i])),2,Scalar(0,0,255),-1);
    imwrite("result/task2_fit/residuals.png",resPlot);

    cap.release();
    writer.release();
    return 0;
}