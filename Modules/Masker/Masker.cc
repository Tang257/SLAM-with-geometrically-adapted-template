/*
 * This file is part of SD-DefSLAM
 * Copyright (C) 2020 Juan J. Gómez Rodríguez, Jose Lamarca Peiro, J. Morlana,
 *                    Juan D. Tardós and J.M.M. Montiel, University of Zaragoza
 *
 * This software is for internal use in the EndoMapper project.
 * Not to be re-distributed.
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <istream>

#include "Masker.h"
#include "BorderMask.h"
#include "BrightMask.h"
#include "CnnSegmentation.h"

namespace defSLAM
{

    void Masker::loadFromTxt(std::string path)
    {
        std::cout << std::endl
                  << "Loading filters: " << path << std::endl;
        //Open filter file
        std::ifstream filterFile(path);

        if (filterFile.is_open())
        {
            std::string line;
            while (getline(filterFile, line))
            {
                std::istringstream ss(line);

                //Get filter name
                std::string name;
                ss >> name;

                if (name == "CNN")
                {
                    CnnSegmentation *cnn = new CnnSegmentation();
                    ss >> name;
                    cnn->loadModel(name);
                    std::unique_ptr<Filter> f(cnn);

                    addFilter(f);
                }
                else if (name == "BorderFilter")
                {
                    std::string rb, re, cb, ce, th;
                    ss >> rb >> re >> cb >> ce >> th;
                    std::unique_ptr<Filter> f(new BorderMask(stoi(rb), stoi(re), stoi(cb), stoi(ce), stoi(th)));
                    addFilter(f);
                }
                else if (name == "BrightFilter")
                {
                    std::string thLo;
                    ss >> thLo;
                    std::unique_ptr<Filter> f(new BrightMask(stoi(thLo)));
                    addFilter(f);
                }
            }

            filterFile.close();
        }
    }

    void Masker::loadRindnetFromTxt(std::string path){
        std::cout << std::endl
                  << "Loading Rindnet model files: " << path << std::endl;
        
        RindNetModelPath_ = path;
    }

    void Masker::loadSampleRatioFromTxt(double ratio){
        std::cout << std::endl
                  << "Loading Sample ratio from files: "<< ratio <<std::endl;
        sample_ratio = ratio;
    }

    void Masker::loadContourNumFromTxt(int num){
        std::cout << std::endl
                  << "Loading Contour Num from files: "<< num <<std::endl;
        contour_num = num;
    }
    
    void Masker::loadContourSampleNumFromTxt(int num){
        std::cout << std::endl
                  << "Loading Contour Sample Num from files: "<< num <<std::endl;
        contour_sample_num = num;
    }

    void Masker::addFilter(std::unique_ptr<Filter> &f)
    {
        filters_.push_back(std::move(f));
    }

    void Masker::deleteFilter(size_t idx)
    {
        filters_.erase(filters_.begin() + idx);
    }

    cv::Mat Masker::mask(const cv::Mat &im)
    {
        //Generates an empty mask (all values set to 0)
        cv::Mat mask(im.rows, im.cols, CV_8U, cv::Scalar(255));

        //Apply each filter
        for (auto &f : filters_)
        {
            cv::bitwise_and(mask, f->generateMask(im), mask);
        }

        return mask;
    }

    void Masker::contourMask(const cv::Mat& im, std::vector<std::vector<cv::Point>> &contours){
        //Generates an empty mask (all values set to 0)
        cv::Mat mask(im.rows, im.cols, CV_8U, cv::Scalar(255));

        //extract contours
        CnnSegmentation *cnn = new CnnSegmentation();
        auto start_all = std::chrono::steady_clock::now(); //开始时间
        cnn->loadRindNetModel(RindNetModelPath_);
        auto end_all = std::chrono::steady_clock::now(); //开始时间
        auto duration_whole = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
        std::cout<<"model_load time: "<<duration_whole.count()<<std::endl;
        std::cout<<"loaded"<<std::endl;
        cv::Mat contour_im = cnn->extractContour(im);
        auto end2_all = std::chrono::steady_clock::now(); //开始时间
        auto duration2_whole = std::chrono::duration_cast<std::chrono::milliseconds>(end2_all - end_all);
        std::cout<<"model_extract time: "<<duration2_whole.count()<<std::endl;
        std::cout<<"extracted"<<std::endl;
        cnn->contourSamples(contour_im, contours, sample_ratio, contour_num, contour_sample_num);
    }

    std::string Masker::printFilters()
    {
        std::string msg("List of filters (" + std::to_string(filters_.size()) + "):\n");

        for (auto &f : filters_)
        {
            msg += "\t-" + f->getDescription() + "\n";
        }

        return msg;
    }
} // namespace defSLAM