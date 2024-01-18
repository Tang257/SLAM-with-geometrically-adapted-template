/*
 * This file is part of SD-DefSLAM
 * Copyright (C) 2020 Juan J. Gómez Rodríguez, Jose Lamarca Peiro, J. Morlana,
 *                    Juan D. Tardós and J.M.M. Montiel, University of Zaragoza
 *
 * This software is for internal use in the EndoMapper project.
 * Not to be re-distributed.
 */

#ifndef DEFORMABLESLAM_MASKER_H
#define DEFORMABLESLAM_MASKER_H

#include<opencv2/core/core.hpp>

#include "Filter.h"

namespace defSLAM {
    class Masker {
        /*
         * This class colects a set of filter to generate a global mask from all of them
         */
    public:
        Masker(){};

        /*
         * Load filters from a .txt file with the following format:
         * <filterName> <param_1> <param_2>
         */
        void loadFromTxt(std::string path);
        void loadRindnetFromTxt(std::string path);
        void loadSampleRatioFromTxt(double ratio);
        void loadContourNumFromTxt(int num);
        void loadContourSampleNumFromTxt(int num);

        /*
         * Adds a filter to the masker
         */
        void addFilter(std::unique_ptr<Filter>& f);

        /*
         * Removes the filter at pos idx
         */
        void deleteFilter(size_t idx);

        /*
         * Applies all filters stored and generates a global mask
         */
        cv::Mat mask(const cv::Mat& im);
        void contourMask(const cv::Mat& im, std::vector<std::vector<cv::Point>> &contours);

        std::string printFilters();

    private:
        std::vector<std::unique_ptr<Filter>> filters_;
        std::string RindNetModelPath_;
        double sample_ratio;
        int contour_num;
        int contour_sample_num;
    };
}

#endif //DEFORMABLESLAM_MASKER_H
