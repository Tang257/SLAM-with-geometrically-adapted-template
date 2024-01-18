/*
 * This file is part of SD-DefSLAM
 * Copyright (C) 2020 Juan J. Gómez Rodríguez, Jose Lamarca Peiro, J. Morlana,
 *                    Juan D. Tardós and J.M.M. Montiel, University of Zaragoza
 *
 * This software is for internal use in the EndoMapper project.
 * Not to be re-distributed.
 */

#include "CnnSegmentation.h"

#include <iostream>

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

namespace defSLAM {

    void CnnSegmentation::loadModel(std::string path) {
        std::cout<<"Loading model from "<<path<<std::endl;
        this->model_ = torch::jit::load(path);
        std::cout << "Loaded model at " << path << std::endl;
    }

    void CnnSegmentation::loadRindNetModel(std::string path){
        std::cout<<"Loading RindNet-model form "<<path<<std::endl;
        this->rindNetModel_ = torch::jit::load(path);
        std::cout<<"Loaded RindNet-model at "<<path<<std::endl;
    }

    cv::Mat CnnSegmentation::forward(const cv::Mat &im) {
        // //Resize image if necessary
        // cv::Mat resizedIm = this->checkImageSize(im);

        // //Preprocess resized image
        // cv::Mat processedIm = this->preprocessMat(resizedIm);

        // //Convert Mat to torch
        // auto cnnInput = this->convertFromMatToTorch(processedIm);

        // //Call neural network
        // auto cnnOutput = model_.forward(cnnInput).toTensor();

        // //Convert back to Mat
        // cv::Mat rawMask = this->convertFromTorchToMat(cnnOutput);

        // //Postprocess image
        // cv::Mat mask = this->postprocessMat(rawMask);

        // //Resize to match the original size
        // cv::resize(mask, mask, im.size());

        //Resize image if necessary
        cv::Mat resizedIm = this->checkImageSize(im);

        //Preprocess resized image
        cv::Mat processedIm = this->preprocessMat(resizedIm);

        //Convert Mat to torch
        auto cnnInput = this->convertFromMatToTorch(processedIm);

        std::cout<<"CNN Before"<<std::endl;

        //Call neural network
        auto cnnOutput = model_.forward(cnnInput).toTensor();
        std::cout<<"CNN 2"<<std::endl;

        //Convert back to Mat
        cv::Mat rawMask = this->convertFromTorchToMat(cnnOutput);

        std::cout<<"CNN 3"<<std::endl;

        //Postprocess image
        cv::Mat mask = this->postprocessMat(rawMask);

        //Resize to match the original size
        cv::resize(mask, mask, im.size());

        return mask.clone();
    }

    cv::Mat CnnSegmentation::extractContour(const cv::Mat& im){
        // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/im_source.png", im);
        cv::Mat resizedIm = this->checkImageSize(im);
        cv::Mat processedIm = this->preprocessMat(resizedIm); // // 这里cnnSegmentation所给的mean,std和RINDNet所给的mean_,std是一样的
        auto RindNetInput = this->convertFromMatToTorch(processedIm);
        auto start_all = std::chrono::steady_clock::now(); //开始时间
        torch::IValue RindNetOutput = rindNetModel_.forward(RindNetInput);
        auto end_all = std::chrono::steady_clock::now(); //开始时间
        auto duration_whole = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
        std::cout<<"model_inference time: "<<duration_whole.count()<<std::endl;
        torch::Tensor depth_output = RindNetOutput.toTuple()->elements()[1].toTensor();
        torch::Tensor normal_output = RindNetOutput.toTuple()->elements()[2].toTensor();
        // 得到RindNetOutput中的第二个变量到depth_map中
        cv::Mat depth_map = this->convertFromTorchToMat(depth_output);
        cv::Mat normal_map = this->convertFromTorchToMat(normal_output);
        cv::Mat depth_mask = this->postprocessMat(depth_map);
        cv::Mat normal_mask = this->postprocessMat(normal_map);
        cv::resize(depth_mask, depth_mask, im.size());
        cv::resize(normal_mask, normal_mask, im.size());
        return this->postprocessContour(depth_mask, normal_mask);
    }

    cv::Mat CnnSegmentation::checkImageSize(const cv::Mat &im) {
        cv::Size newSize = im.size(), originalSize = im.size();

        float fractional;
        if (modf(float(originalSize.width) / (float) SIZE_DIV_, &fractional) > 0.f)
            newSize.width = originalSize.width / SIZE_DIV_ * SIZE_DIV_;

        if (modf(float(originalSize.height) / (float) SIZE_DIV_, &fractional) > 0.f)
            newSize.height = originalSize.height / SIZE_DIV_ * SIZE_DIV_;

        cv::Mat resizedIm;
        cv::resize(im, resizedIm, newSize);

        return resizedIm.clone();
    }

    cv::Mat CnnSegmentation::preprocessMat(const cv::Mat &im) {
        cv::Mat out;

        im.copyTo(out);
        out.convertTo(out, CV_32FC3);

        //Subtract mean
        cv::subtract(out, mean_, out);

        //Divide by std
        cv::divide(out, std_, out);

        return out.clone();
    }

    std::vector<torch::jit::IValue> CnnSegmentation::convertFromMatToTorch(const cv::Mat &im) {
        auto input_tensor = torch::from_blob(im.data, {im.rows, im.cols, im.channels()});
        input_tensor = input_tensor.permute({2, 0, 1});

        auto unsqueezed = input_tensor.unsqueeze(0);

        std::vector <torch::jit::IValue> torchVector;
        torchVector.push_back(unsqueezed);

        return torchVector;
    }

    cv::Mat CnnSegmentation::convertFromTorchToMat(at::Tensor &tensor) {
        cv::Mat mat(tensor.size(2), tensor.size(3), CV_32F, (void *) tensor.data<float>());
        return mat.clone();
    }

    cv::Mat CnnSegmentation::depthTorchToMat(at::Tensor &tensor){
        cv::Mat mat(tensor[1].size(2), tensor[1].size(3), CV_32F, (void *) tensor.data<float>());
        return mat.clone();
    }
    cv::Mat CnnSegmentation::normalTorchToMat(at::Tensor &tensor){
        cv::Mat mat(tensor[2].size(2), tensor[2].size(3), CV_32F, (void *) tensor.data<float>());
        return mat.clone();
    }

    cv::Mat CnnSegmentation::postprocessMat(const cv::Mat &im) {
        cv::Mat out;

        cv::threshold(im, out, 0.1f, 255, cv::THRESH_BINARY);
        out.convertTo(out, CV_8U);

        return out.clone();
    }

    cv::Mat CnnSegmentation::postprocessContour(const cv::Mat &depth_im, const cv::Mat &normal_im){
        std::clock_t start = std::clock();
        cv::Mat img1_2 = cv::Mat::zeros(depth_im.size(), CV_8U);
        depth_im(cv::Rect(32, 8, depth_im.cols-32-12, depth_im.rows-16-4)).copyTo(img1_2(cv::Rect(32, 8, depth_im.cols-32-12, depth_im.rows-16-4)));
        cv::Mat img2_2 = cv::Mat::zeros(normal_im.size(), CV_8U);
        normal_im(cv::Rect(32, 8, depth_im.cols-32-12, depth_im.rows-16-4)).copyTo(img2_2(cv::Rect(32, 8, depth_im.cols-32-12, depth_im.rows-16-4)));
        // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/depth_contour_"+std::to_string(start)+".jpg", depth_im);
        // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/normal_contour_"+std::to_string(start)+".jpg", normal_im);

        // TODO: 此处5是阈值，需要调整
        cv::threshold(img1_2, img1_2, 5, 255, cv::THRESH_BINARY);
        cv::threshold(img2_2, img2_2, 5, 255, cv::THRESH_BINARY);

        // 二值化
        cv::Mat img3 = cv::Mat(depth_im.size(), CV_8U, cv::Scalar(0));
        cv::bitwise_or(img1_2, img2_2, img3);
        // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/depth_normal_contour_"+std::to_string(start)+".jpg", img3);

        // 对二值化图像进行处理去噪
        cv::Mat img5 = cv::Mat(depth_im.size(), CV_8U, cv::Scalar(0));
        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(img3, img5, cv::MORPH_OPEN, kernal);
        // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/edgeWithNoise_"+std::to_string(start)+".jpg", img5);

        // 提取骨架线
        cv::Mat edge_skeleton = skeletonProcess(img5);
        // imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/edge_skeleton_"+std::to_string(start)+".jpg", edge_skeleton);

        return edge_skeleton.clone();
    }

    void CnnSegmentation::contourSamples(const cv::Mat& im, std::vector<std::vector<cv::Point>> &contours_sampled, float sample_rate, int contour_num, int sample_max_num){
        cv::imwrite("im_contours.jpg", im);
        // 寻找轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(im, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

        // 选择最大长度的contour_num个contour
        std::vector<int> contours_length;
        for (int i = 0; i < contours.size(); i++) {
            contours_length.push_back(contours[i].size());
        }
        std::sort(contours_length.begin(), contours_length.end());

        if (contours.size() < contour_num){
            contour_num = contours.size();
        }
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() < contours_length[contours_length.size()-contour_num]){
                //移除当前contour
                contours.erase(contours.begin()+i);
                i--;
            }
        }

        int sum_length = 0;
        for (int i = 0; i < contours.size(); i++){
            sum_length += contours[i].size();
        }

        std::cout<<"sample_max_num: "<<sample_max_num<<" sample_rate: "<<sample_rate<<std::endl;

        if (sample_max_num > 0){
            for (int i = 0; i < contours.size(); i++){
                int sample_num = round((float)sample_max_num*((float)contours[i].size()/(float)sum_length));
                if (sample_num == 0){
                    continue;
                }else{
                    float interval = (float)contours[i].size() / (float)sample_num;
                    std::vector<cv::Point> contour_sampled;
                    
                    for (int j = 0; j < sample_num; j++){
                        float seg_ratio = 0.5;
                        int index = j * interval + seg_ratio*interval;
                        contour_sampled.push_back(contours[i][index]);
                    }
                    contours_sampled.push_back(contour_sampled);   
                }
            }
        }else{
            for (int i = 0; i < contours.size(); i++){
                int sample_num = contours[i].size() * sample_rate;
                if (sample_num == 0){
                    continue;
                }else{
                    float interval = contours[i].size() / sample_num;
                    std::vector<cv::Point> contour_sampled;
                    
                    for (int j = 0; j < sample_num; j++){
                        float seg_ratio = 0.5;
                        int index = j * interval + seg_ratio*interval;
                        contour_sampled.push_back(contours[i][index]);
                    }
                    contours_sampled.push_back(contour_sampled);   
                }
            }
        }

        cv::Mat im_contours = cv::Mat::zeros(im.size(), CV_8UC1);
        for (int i = 0; i < contours_sampled.size(); i++){
            for (int j = 0; j < contours_sampled[i].size(); j++){
                cv::circle(im_contours, contours_sampled[i][j], 2, cv::Scalar(255), 1);
            }
        }
        std::clock_t start = std::clock();
        // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/im_samples_"+std::to_string(start)+".jpg", im_contours);

    }

    cv::Mat CnnSegmentation::skeletonProcess(cv::Mat &sourceBinaryImage){
        for(int i = 0; i < sourceBinaryImage.rows; i++)
        {
            for(int j = 0; j < sourceBinaryImage.cols; j++)
                if((int)sourceBinaryImage.at<uchar>(i, j) >= 5)
                    sourceBinaryImage.at<uchar>(i, j) = 255;
                else
                    sourceBinaryImage.at<uchar>(i, j) = 0;
        }

        MatrixXi I_source(sourceBinaryImage.rows, sourceBinaryImage.cols);
        
        for(int i = 0; i < sourceBinaryImage.rows; i++)
        {
            for(int j = 0; j < sourceBinaryImage.cols; j++)
                if((int)sourceBinaryImage.at<uchar>(i, j) >= 127)
                    I_source(i, j) = 1;
                else
                    I_source(i, j) = 0;
        }
        MatrixXi I = MatrixXi::Zero(I_source.rows()+6, I_source.cols()+6);
        I.block(3, 3, I_source.rows(), I_source.cols()) = I_source;

        MatrixXi DeleteMatrix = MatrixXi::Zero(I.rows(), I.cols());

        MatrixXi Rt_a(3, 4), Rt_b(4, 3); // 定义两个Restoring templates
        MatrixXi Cdt(4, 4); // 定义一个Compulsory deletion template
        Rt_a<<3, 3, 3, 3, 
            0, 2, 1, 0, 
            3, 1, 3, 3;

        Rt_b<<3, 0, 3, 
            3, 2, 1, 
            3, 1, 3, 
            3, 0, 3;

        Cdt<<3, 0, 3, 4, 
            0, 2, 1, 4, 
            3, 1, 3, 4, 
            4, 4, 4, -1;
        
        // 定义的Extra deletion templates 10个
        MatrixXi Edt_d(5, 4), Edt_e(4, 5), Edt_f(4, 5), Edt_g(4, 5), Edt_h(4, 5), Edt_i(5, 4), Edt_j(5, 4), Edt_k(5, 4), Edt_l(3, 3), Edt_m(3, 3);
        Edt_d<<0, 0, 0, -1, 
            0, 1, 0, 3, 
            0, 2, 1, 5,  // x = 3, G1 = 4, G2 = 5
            3, 1, 1, 5, 
            3, 4, 4, -1;

        Edt_e<<0, 0, 0, 3, 3, 
            0, 1, 2, 1, 5, 
            0, 0, 1, 1, 5, 
            -1, 3, 4, 4, -1;

        Edt_f<<-1, 3, 6, 6, -1,  // E1 = 6, E2 = 7
            0, 0, 1, 1, 7, 
            0, 1, 2, 1, 7, 
            0, 0, 0, 3, 3;

        Edt_g<<-1, 6, 6, 3, -1, 
            7, 1, 1, 0, 0, 
            7, 1, 2, 1, 0, 
            3, 3, 0, 0, 0;

        Edt_h<<3, 3, 0, 0, 0, 
            6, 1, 2, 1, 0, 
            6, 1, 1, 0, 0, 
            -1, 7, 7, 3, -1;

        Edt_i<<-1, 0, 0, 0, 
            3, 0, 1, 0, 
            6, 1, 2, 0, 
            6, 1, 1, 3, 
            -1, 7, 7, 3;

        Edt_j<<3, 6, 6, -1, 
            3, 1, 1, 7, 
            0, 2, 1, 7, 
            0, 1, 0, 3, 
            0, 0, 0, -1;

        Edt_k<<-1, 6, 6, 3, 
            7, 1, 1, 3, 
            7, 1, 2, 0, 
            3, 0, 1, 0, 
            -1, 0, 0, 0;

        Edt_l<<1, 1, 0, 
            0, 2, 1, 
            0, 0, 1;

        Edt_m<<0, 1, 1, 
            1, 2, 0, 
            1, 0, 0;

        bool flag = false;
        int itera = 0;
        do{
            flag = false;
            for (int r = 3; r < I.rows()-3; r++){
                for (int c = 3; c < I.cols()-3; c++){
                    int P = I(r, c);  //是否需要减1
                    //如果P是前景点
                    if (P == 1){
                        // 取P的8邻域
                        int PN[8];
                        PN[7] = I(r-1, c-1);PN[0] = I(r-1, c);PN[1] = I(r-1, c+1);
                        PN[6] = I(r, c-1);                    PN[2] = I(r, c+1);
                        PN[5] = I(r+1, c-1);PN[4] = I(r+1, c);PN[3] = I(r+1, c+1);

                        //取P的8领域的外围12领域
                        int PA[12];
                                            PA[0] = I(r-2, c-1);PA[1] = I(r-2, c);PA[2] = I(r-2, c+1);
                        PA[11] = I(r-1, c-2);                                                            PA[3] = I(r-1, c+2);
                        PA[10] = I(r, c-2);                                                              PA[4] = I(r, c+2);
                        PA[9] = I(r+1, c-2);                                                             PA[5] = I(r+1, c+2);
                                            PA[8] = I(r+2, c-1);PA[7] = I(r+2, c);PA[6] = I(r+2, c+1);
                        
                        // 计算A(P)
                        int AP = 0; 
                        for (int a = 1; a <= 4; a++){
                            AP += !PN[2*a-1-1]*PN[2*a-1]+!PN[2*a-1]*PN[(2*a+1-1)%8]; // EQ(7)
                        }

                        // 计算B(P) P的8领域中前景点的个数
                        int BP = 0;
                        for (int a = 0; a < 8; a++){
                            if (PN[a] == 1){
                                BP += 1; // EQ(9)
                            }
                        }
                        
                        if (AP==1 && BP>=2 && BP<=6){
                            // 检查P的20领域和restoring template是否有匹配的
                            // 检查是否不和Restoring_template的任意一个template匹配
                            bool notMatchRta = true, notMatchRtb = true;
                            if (Rt_a(1, 0)==PN[6] && Rt_a(1, 2)==PN[2] && Rt_a(2, 1)==PN[4] && Rt_a(1, 3)==PA[4]){
                                notMatchRta = false;
                            } // 和template_a匹配
                            if (Rt_b(0, 1)==PN[0] && Rt_b(1, 2)==PN[2] && Rt_b(2, 1)==PN[4] && Rt_b(3, 1)==PA[7]){ // 和template_b匹配
                                notMatchRtb = false;
                            }

                            // 检查是否和Compulsory_deletion_template匹配
                            bool matchCdt = false;
                            int num_of_t = 0;
                            for (int a = 3; a <= 8; a++){ //changed
                                if (PA[a] == 1){
                                    num_of_t += 1;
                                }
                            }
                            if (Cdt(0, 1)==PN[0] && Cdt(1, 2)==PN[2] && Cdt(2, 1)==PN[4] && Cdt(1, 0)==PN[6] && num_of_t>=2){
                                matchCdt = true;
                            }

                            if (notMatchRta && notMatchRtb){
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }
                            if (notMatchRta!=notMatchRtb && matchCdt){
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }

                        }else if(AP==2 && BP >= 4 && BP <= 5){
                            // 检查是否有和extra deletion templates相匹配的模板
                            bool matchEdt_d = false;

                            // 与Edt_d
                            if (PA[0]==Edt_d(0, 0) && PA[1]==Edt_d(0, 1) && PA[2]==Edt_d(0, 2) && 
                                PN[7]==Edt_d(1, 0) && PN[0]==Edt_d(1, 1) && PN[1]==Edt_d(1, 2) &&
                                PN[6]==Edt_d(2, 0) && PN[2]==Edt_d(2, 2) &&
                                PN[4]==Edt_d(3, 1) && PN[3]==Edt_d(3, 2) && 
                                PA[4]==PA[5] && PA[6]==PA[7] && (PA[4]+PA[5]+PA[6]+PA[7])>=1){
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PA[11]==Edt_e(0, 0) && PN[7]==Edt_e(0, 1) && PN[0]==Edt_e(0, 2) &&
                                    PA[10]==Edt_e(1, 0) && PN[6]==Edt_e(1, 1) && PN[2]==Edt_e(1, 3) && 
                                    PA[9]==Edt_e(2, 0) && PN[5]==Edt_e(2, 1) && PN[4]==Edt_e(2, 2) && PN[3]==Edt_e(2, 3) && 
                                    PA[4]==PA[5] && PA[6]==PA[7] && (PA[4]+PA[5]+PA[6]+PA[7])>=1){
                                // 与Edt_e
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PA[11]==Edt_f(1, 0) && PN[7]==Edt_f(1, 1) && PN[0]==Edt_f(1, 2) && PN[1]==Edt_f(1, 3) && 
                                    PA[10]==Edt_f(2, 0) && PN[6]==Edt_f(2, 1) && PN[2]==Edt_f(2, 3) && 
                                    PA[9]==Edt_f(3, 0) && PN[5]==Edt_f(3, 1) && PN[4]==Edt_f(3, 2) && 
                                    PA[3]==PA[4] && PA[2]==PA[1]){
                                // 与Edt_f
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PN[7]==Edt_g(1, 1) && PN[0]==Edt_g(1, 2) && PN[1]==Edt_g(1, 3) && PA[3]==Edt_g(1, 4) && 
                                    PN[6]==Edt_g(2, 1) && PN[2]==Edt_g(2, 3) && PA[4]==Edt_g(2, 4) && 
                                    PN[4]==Edt_g(3, 2) && PN[3]==Edt_g(3, 3) && PA[5]==Edt_g(3, 4) && 
                                    PA[0]==PA[1] && PA[10]==PA[11]){
                                // 与Edt_g
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PN[0]==Edt_h(0, 2) && PN[1]==Edt_h(0, 3) && PA[3]==Edt_h(0, 4) && 
                                    PN[6]==Edt_h(1, 1) && PN[2]==Edt_h(1, 3) && PA[4]==Edt_h(1, 4) && 
                                    PN[5]==Edt_h(2, 1) && PN[4]==Edt_h(2, 2) && PN[3]==Edt_h(2, 3) && PA[5]==Edt_h(2, 4) && 
                                    PA[9]==PA[10] && PA[7]==PA[8]){
                                // 与Edt_h
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PA[0]==Edt_i(0, 1) && PA[1]==Edt_i(0, 2) && PA[2]==Edt_i(0, 3) && 
                                    PN[7]==Edt_i(1, 1) && PN[0]==Edt_i(1, 2) && PN[1]==Edt_i(1, 3) && 
                                    PN[6]==Edt_i(2, 1) && PN[2]==Edt_i(2, 3) && 
                                    PN[5]==Edt_i(3, 1) && PN[4]==Edt_i(3, 2) && 
                                    PA[9]==PA[10] && PA[7]==PA[8]){
                                // 与Edt_i
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PN[0]==Edt_j(1, 1) && PN[1]==Edt_j(1, 2) && 
                                    PN[6]==Edt_j(2, 0) && PN[2]==Edt_j(2, 2) && 
                                    PN[5]==Edt_j(3, 0) && PN[4]==Edt_j(3, 1) && PN[3]==Edt_j(3, 2) && 
                                    PA[8]==Edt_j(4, 0) && PA[7]==Edt_j(4, 1) && PA[6]==Edt_j(4, 2) && 
                                    PA[1]==PA[2] && PA[3]==PA[4]){
                                // 与Edt_j
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PN[7]==Edt_k(1, 1) && PN[0]==Edt_k(1, 2) && 
                                    PN[6]==Edt_k(2, 1) && PN[2]==Edt_k(2, 3) && 
                                    PN[5]==Edt_k(3, 1) && PN[4]==Edt_k(3, 2) && PN[3]==Edt_k(3, 3) && 
                                    PA[8]==Edt_k(4, 1) && PA[7]==Edt_k(4, 2) && PA[6]==Edt_k(4, 3) && // changed
                                    PA[0]==PA[1] && PA[10]==PA[11]){
                                // 与Edt_k
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PN[7]==Edt_l(0, 0) && PN[0]==Edt_l(0, 1) && PN[1]==Edt_l(0, 2) && 
                                    PN[6]==Edt_l(1, 0) && PN[2]==Edt_l(1, 2) && 
                                    PN[5]==Edt_l(2, 0) && PN[4]==Edt_l(2, 1) && PN[3]==Edt_l(2, 2)){
                                // 与Edt_l
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }else if(PN[7]==Edt_m(0, 0) && PN[0]==Edt_m(0, 1) && PN[1]==Edt_m(0, 2) && 
                                    PN[6]==Edt_m(1, 0) && PN[2]==Edt_m(1, 2) && 
                                    PN[5]==Edt_m(2, 0) && PN[3]==Edt_m(2, 1) && PN[3]==Edt_m(2, 2)){
                                // Edt_m
                                DeleteMatrix(r, c) = 1;
                                flag = true;
                            }
                        }
                    }
                }
            }
            // 从I中对应删除DeleteMatrix中的元素
            for(int r=0; r<I.rows(); r++){
                for(int c=0; c<I.cols(); c++){
                    if(DeleteMatrix(r, c)==1){
                        I(r, c) = 0;
                    }
                }
            }
            DeleteMatrix = MatrixXi::Zero(I.rows(), I.cols());
            I_source = I.block(3, 3, I.rows()-6, I.cols()-6);
            // 保存 TODO
            itera++;
        }while(flag);

        // 检查是否保持单像素宽度
        for (int r=3; r < I.rows()-3; r++){
            for (int c=3; c < I.cols()-3; c++){
                if (I(r, c)==1){
                    // 提取8领域
                    int PN[8];
                    PN[7] = I(r-1, c-1);PN[0] = I(r-1, c);PN[1] = I(r-1, c+1);
                    PN[6] = I(r, c-1);                    PN[2] = I(r, c+1);
                    PN[5] = I(r+1, c-1);PN[4] = I(r+1, c);PN[3] = I(r+1, c+1);

                    if (PN[0]*PN[2]*!PN[5]==1 || PN[2]*PN[4]*!PN[7]==1 || PN[4]*PN[6]*!PN[1]==1 || PN[6]*PN[0]*!PN[3]==1){
                        I(r, c) = 0;
                    }
                }
            }
        }
        // 保存结果
        I_source = I.block(3, 3, I.rows()-6, I.cols()-6);
        cv::Mat I_result = cv::Mat(I_source.rows(), I_source.cols(), CV_8UC1);
        for(int r=0; r<I_source.rows(); r++){
            for(int c=0; c<I_source.cols(); c++){
                if(I_source(r, c)==1){
                    I_result.at<uchar>(r, c) = 255;
                }else{
                    I_result.at<uchar>(r, c) = 0;
                }
            }
        }

        return I_result;
    }

    std::string CnnSegmentation::getDescription(){
        return std::string("Tool segmentation with CNN");
    }
}