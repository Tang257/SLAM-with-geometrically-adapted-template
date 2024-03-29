/*
 * This file is part of SD-DefSLAM
 * Copyright (C) 2020 Juan J. Gómez Rodríguez, Jose Lamarca Peiro, J. Morlana,
 *                    Juan D. Tardós and J.M.M. Montiel, University of Zaragoza
 *
 * This software is for internal use in the EndoMapper project.
 * Not to be re-distributed.
 */


#include <SettingsLoader.h>
#include <iostream>

namespace defSLAM
{
    SettingsLoader::SettingsLoader()
        : SettingsLoader(DEFAULTCALIBRATION) // MACRO DEFINED IN CMakeLists.txt
    {
    }

    SettingsLoader::SettingsLoader(const std::string &strSettingsFile)
    {
        using namespace std;
        cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fSettings.isOpened())
        {
            std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
            exit(-1);
        }
        // Deformation tracking
        regLap_ = fSettings["Regularizer.laplacian"];
        regInex_ = fSettings["Regularizer.Inextensibility"];
        regTemp_ = fSettings["Regularizer.temporal"];
        localZoneSize_ = fSettings["Regularizer.LocalZone"];
        float saveResults = fSettings["Viewer.SaveResults"];

        saveResults_ = bool(uint(saveResults));

        // Tracking
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(K_);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(distCoef_);

        bf_ = fSettings["Camera.bf"];

        fps_ = fSettings["Camera.fps"];
        if (fps_ == 0)
            fps_ = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        minFrames_ = 0;
        maxFrames_ = fps_;

        cout << endl
             << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps_ << endl;

        int nRGB = fSettings["Camera.RGB"];
        RGB_ = nRGB;

        if (RGB_)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters
        nFeatures_ = fSettings["ORBextractor.nFeatures"];
        fScaleFactor_ = fSettings["ORBextractor.scaleFactor"];
        nLevels_ = fSettings["ORBextractor.nLevels"];
        fIniThFAST_ = fSettings["ORBextractor.iniThFAST"];
        fMinThFAST_ = fSettings["ORBextractor.minThFAST"];

        // Local Mapping
        pointsToTemplate_ = fSettings["LocalMapping.pointsToTemplate"];
        chiLimit_ = fSettings["LocalMapping.chiLimit"];
        schwarpReg_ = fSettings["LocalMapping.Schwarp.Regularizer"];
        bendingReg_ = fSettings["LocalMapping.Bending"];

        // Viewer
        if (fps_ < 1)
            fps_ = 30;
        T_ = 1e3 / 30;

        imageWidth_ = fSettings["Camera.width"];
        imageHeight_ = fSettings["Camera.height"];
        if (imageWidth_ < 1 || imageHeight_ < 1)
        {
            imageWidth_ = 640;
            imageWidth_ = 480;
        }

        viewpointX_ = fSettings["Viewer.ViewpointX"];
        viewpointY_ = fSettings["Viewer.ViewpointY"];
        viewpointZ_ = fSettings["Viewer.ViewpointZ"];
        viewpointF_ = fSettings["Viewer.ViewpointF"];

        // Map Drawer
        keyFrameSize_ = fSettings["Viewer.KeyFrameSize"];
        keyFrameLineWidth_ = fSettings["Viewer.KeyFrameLineWidth"];
        graphLineWidth_ = fSettings["Viewer.GraphLineWidth"];
        pointSize_ = fSettings["Viewer.PointSize"];
        cameraSize_ = fSettings["Viewer.CameraSize"];
        cameraLineWidth_ = fSettings["Viewer.CameraLineWidth"];

        float debugPoints = fSettings["Debug.bool"];

        debugPoints_ = bool(uint(debugPoints));

        filterPath_ = fSettings["Filters.file"].string();
        rindNetPath_ = fSettings["RindNet.Model"].string();
    }

    int SettingsLoader::getFPS() const
    {
        return fps_;
    }

    int SettingsLoader::getCameraWidth() const
    {
        return imageWidth_;
    }
    int SettingsLoader::getCameraHeight() const
    {
        return imageHeight_;
    }
    float SettingsLoader::getViewPointX() const
    {
        return viewpointX_;
    }
    float SettingsLoader::getViewPointY() const
    {
        return viewpointY_;
    }
    float SettingsLoader::getViewPointZ() const
    {
        return viewpointZ_;
    }
    float SettingsLoader::getViewPointF() const
    {
        return viewpointF_;
    }
    bool SettingsLoader::getSaveResults() const
    {
        return saveResults_;
    }

    float SettingsLoader::getkeyFrameSize() const
    {
        return keyFrameSize_;
    }
    float SettingsLoader::getkeyFrameLineWidth() const
    {
        return keyFrameLineWidth_;
    }
    float SettingsLoader::getpointSize() const
    {
        return pointSize_;
    }
    float SettingsLoader::getcameraSize() const
    {
        return cameraSize_;
    }
    float SettingsLoader::getcameraLineWidth() const
    {
        return cameraLineWidth_;
    }
    float SettingsLoader::getgraphLineWidth() const
    {
        return graphLineWidth_;
    }

    cv::Mat SettingsLoader::getK() const
    {
        return K_;
    }
    cv::Mat SettingsLoader::getdistCoef() const
    {
        return distCoef_;
    }
    float SettingsLoader::getbf() const
    {
        return bf_;
    }
    int SettingsLoader::getminFrames() const
    {
        return minFrames_;
    }
    int SettingsLoader::getmaxFrames() const
    {
        return maxFrames_;
    }
    int SettingsLoader::getRGB() const
    {
        return RGB_;
    }
    int SettingsLoader::getnFeatures() const
    {
        return nFeatures_;
    }
    float SettingsLoader::getfScaleFactor() const
    {
        return fScaleFactor_;
    }
    int SettingsLoader::getnLevels() const
    {
        return nLevels_;
    }
    int SettingsLoader::getfIniThFAST() const
    {
        return fIniThFAST_;
    }
    int SettingsLoader::getfMinThFAST() const
    {
        return fMinThFAST_;
    }
    float SettingsLoader::getregLap() const
    {
        return regLap_;
    }

    float SettingsLoader::getregTemp() const
    {
        return regTemp_;
    }

    float SettingsLoader::getregStreching() const
    {
        return regInex_;
    }

    float SettingsLoader::getLocalZone() const
    {
        return localZoneSize_;
    }

    float SettingsLoader::getreliabilityThreshold() const
    {
        return reliabilityThreshold_;
    }

    int SettingsLoader::getpointsToTemplate() const
    {
        return pointsToTemplate_;
    }
    float SettingsLoader::getchiLimit() const
    {
        return chiLimit_;
    }
    float SettingsLoader::getschwarpReg() const
    {
        return schwarpReg_;
    }
    float SettingsLoader::getbendingReg() const
    {
        return bendingReg_;
    }
    float SettingsLoader::getfps() const
    {
        return fps_;
    }
    float SettingsLoader::getT() const
    {
        return T_;
    }
    std::string SettingsLoader::getFilterPath() const
    {
        return filterPath_;
    }
    std::string SettingsLoader::getRindNetPath() const
    {
        return rindNetPath_;
    }
    bool SettingsLoader::getDebugPoints() const
    {
        return debugPoints_;
    }
    void SettingsLoader::setSaveResults(const bool save)
    {
        saveResults_ = save;
    }

    void SettingsLoader::setK(const cv::Mat &k)
    {
        k.copyTo(K_);
        K_.convertTo(K_, CV_32F);
    }

    void SettingsLoader::setD(const cv::Mat &D)
    {
        D.copyTo(distCoef_);
        distCoef_.convertTo(distCoef_, CV_32F);
    }

    void SettingsLoader::setbf(const float bf)
    {
        bf_ = bf;
    }

    void SettingsLoader::setCameraWidth(const int w)
    {
        imageWidth_ = w;
    }
    void SettingsLoader::setCameraHeight(const int h)
    {
        imageHeight_ = h;
    }
    void SettingsLoader::setFilterPath(const std::string &s)
    {
        filterPath_ = s;
    }
} // namespace defSLAM