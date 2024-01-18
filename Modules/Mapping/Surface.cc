/**
* This file is part of DefSLAM.
*
* Copyright (C) 2017-2020 Jose Lamarca Peiro <jlamarca at unizar dot es>, J.M.M. Montiel (University
*of Zaragoza) && Shaifali Parashar, Adrien Bartoli (Université Clermont Auvergne)
* For more information see <https://github.com/unizar/DefSLAM>
*
* DefSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DefSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DefSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Surface.h"
#include "DefKeyFrame.h" 

#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<fstream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/lloyd_optimize_mesh_2.h>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;

typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel;
typedef CGAL::Delaunay_mesh_vertex_base_2<kernel> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<kernel> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<kernel, Tds, Itag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;
typedef CDT::Face_handle Face_handle;


namespace BBS
{
  void EvalEigen(bbs_t *bbs, const double *ctrlpts, double *u, double *v,
                 int nval, Eigen::MatrixXd &val, int du, int dv);
}

namespace defSLAM
{
  // Constructor
  Surface::Surface(uint numberOfPoints)
      : nodesDepth_(nullptr), surfacePoints_(numberOfPoints, SurfacePoint()),
        numberofNormals_(0), normalsLimit_(10)
  {
  }

  // Destructor
  Surface::~Surface()
  {
    surfacePoints_.clear();
    if (nodesDepth_)
      delete[] nodesDepth_;
  }

  /***********
  * Save the array with the depth estimated by 
  ************/
  void Surface::saveArray(double *Array, BBS::bbs_t &bbss)
  {
    bbs = bbss;
    if (nodesDepth_)
      delete[] nodesDepth_;

    nodesDepth_ = new double[bbs.nptsu * bbs.nptsv];
    for (int i(0); i < bbs.nptsu * bbs.nptsv; i++)
    {
      nodesDepth_[i] = Array[i];
    }
  }

  /***********
  * Check if enough normals have been estimated
  * to define the surface with Shape-from-normals.
  ************/
  bool Surface::enoughNormals()
  {
    std::cout << "Number Of normals " << numberofNormals_ << " " << this
              << std::endl;
    return (numberofNormals_ >= normalsLimit_);
  }

  // Return normals saved
  uint Surface::getnumberofNormals() { return numberofNormals_; }

  /***********
    * Set a normal saved. In DefSLAM ind makes reference to
    * the keypoint index
    ************/
  void Surface::setNormalSurfacePoint(size_t ind, cv::Vec3f &N)
  {
    // Count normals just once
    if (!surfacePoints_[ind].thereisNormal())
    {
      numberofNormals_++;
    }

    surfacePoints_[ind].setNormal(N);
  }

  // Get the normal of the keypoint with index ind.
  bool Surface::getNormalSurfacePoint(size_t ind, cv::Vec3f &N)
  {
    return surfacePoints_[ind].getNormal(N);
  }

  // Set the 3D position of the keypoint with index ind.
  void Surface::set3DSurfacePoint(size_t ind, cv::Vec3f &x3D)
  {
    surfacePoints_[ind].setDepth(x3D);
  }

  // Get the 3D position of the keypoint with index ind.
  void Surface::get3DSurfacePoint(size_t ind, cv::Vec3f &x3D)
  {
    surfacePoints_[ind].getDepth(x3D);
  }

  // Apply scale to the entire surface. Called after surface
  // registration
  void Surface::applyScale(double s22)
  {
    for (uint i(0); i < surfacePoints_.size(); i++)
    {
      cv::Vec3f x3D;
      surfacePoints_[i].getDepth(x3D);
      cv::Vec3f x3c = s22 * x3D;
      surfacePoints_[i].setDepth(x3c);
    }

    for (int i(0); i < bbs.nptsu * bbs.nptsv; i++)
    {
      nodesDepth_[i] = s22 * nodesDepth_[i];
    }
  }

  void drawTriangulation(CDT triangulation, cv::Mat &image, float umin, float umax, float vmin, float vmax){

    // 将triangulation 里的网格画到triangulation_mesh上
    for (auto it = triangulation.finite_faces_begin(); it != triangulation.finite_faces_end(); it++){
      Point p1 = it->vertex(0)->point();
      Point p2 = it->vertex(1)->point();
      Point p3 = it->vertex(2)->point();
      // std::cout<<p1.x()<<" "<<p1.y()<<" "<<p2.x()<<" "<<p2.y()<<" "<<p3.x()<<" "<<p3.y()<<std::endl;

      float x1, y1, x2, y2, x3, y3;
      std::stringstream ss;
      ss<<p1.x(); ss>>x1; ss.clear();
      ss<<p1.y(); ss>>y1; ss.clear();
      ss<<p2.x(); ss>>x2; ss.clear();
      ss<<p2.y(); ss>>y2; ss.clear();
      ss<<p3.x(); ss>>x3; ss.clear();
      ss<<p3.y(); ss>>y3; ss.clear();

      // std::cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<x3<<" "<<y3<<std::endl;

      int x1_i = (x1 - umin) / (umax - umin) * image.cols;
      int y1_i = (y1 - vmin) / (vmax - vmin) * image.rows;
      int x2_i = (x2 - umin) / (umax - umin) * image.cols;
      int y2_i = (y2 - vmin) / (vmax - vmin) * image.rows;
      int x3_i = (x3 - umin) / (umax - umin) * image.cols;
      int y3_i = (y3 - vmin) / (vmax - vmin) * image.rows;

      cv::line(image, cv::Point(x1_i, y1_i), cv::Point(x2_i, y2_i), cv::Scalar(255), 1);
      cv::line(image, cv::Point(x2_i, y2_i), cv::Point(x3_i, y3_i), cv::Scalar(255), 1);
      cv::line(image, cv::Point(x3_i, y3_i), cv::Point(x1_i, y1_i), cv::Scalar(255), 1);
    }
  }

  // Discretize the surface in xs*ys vertex. It is used to
  // create a mesh of xs columns and ys rows.
  void Surface::getVertex(std::vector<cv::Mat> &NodesSurface, uint xs, uint ys)
  {
    double arrayCU[xs * ys];
    double arrayCV[xs * ys];

    uint us(0);
    double t(0.03);
    // bbs在ShapeFromNormals中被初始化，其中bbs.umax=referenceKeyframe.umax, bbs.umin=referenceKeyframe.umin, bbs.vmax=referenceKeyframe.vmax, bbs.vmin=referenceKeyframe.vmin
    double umaxtemp = bbs.umax; //-0.20;
    double umintemp = bbs.umin; //+0.15;
    double vmaxtemp = bbs.vmax; //-0.20;
    double vmintemp = bbs.vmin; //+0.25;
    // std::cout<<"umaxtemp: "<<umaxtemp<<" umintemp: "<<umintemp<<" vmaxtemp: "<<vmaxtemp<<" vmintemp: "<<vmintemp<<std::endl;
    for (uint x(0); x < xs; x++)
    {
      for (uint j(0); j < ys; j++)
      {
        arrayCU[us] =
            double((umaxtemp - umintemp - 2 * t) * x) / (xs - 1) + (umintemp + t);
        arrayCV[us] =
            double((vmaxtemp - vmintemp - 2 * t) * j) / (ys - 1) + (vmintemp + t);
        // std::cout<<arrayCU[us]<<" "<<arrayCV[us]<<std::endl;
        us++;
      }
    }

    Eigen::MatrixXd Val2(xs * ys, 1);
    // B-样条曲面：可以计算出曲面上的任意点的坐标，nodesDepth在SurfaceRegistration::registerSurfaces()中被赋值为尺度对齐后每个点的深度
    BBS::EvalEigen(&bbs, static_cast<const double *>(nodesDepth_), arrayCU,
                   arrayCV, xs * ys, Val2, 0, 0); // 这里相当于解出每个点对应在三维模板空间中的深度Val2
    NodesSurface.reserve(xs * ys);
    std::ofstream fout("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/getVertex.txt");
    for (uint x(0); x < xs * ys; x++)
    {
      cv::Mat x3D(4, 1, CV_32F);
      x3D.at<float>(0, 0) = arrayCU[x] * Val2(x, 0);
      x3D.at<float>(1, 0) = arrayCV[x] * Val2(x, 0);
      x3D.at<float>(2, 0) = Val2(x, 0);
      x3D.at<float>(3, 0) = 1;
      // std::cout << "x3D: " << x3D << std::endl;
      fout<<x3D.at<float>(0, 0)<<" "<<x3D.at<float>(1, 0)<<" "<<x3D.at<float>(2, 0)<<std::endl;
      NodesSurface.push_back(x3D);
    }
    fout.close();
  }

  void Surface::getAdaptiveVertex(std::vector<cv::Mat> &NodesSurface, std::vector<std::vector<int>> &facets, uint xs, uint ys, uint width, uint height, std::vector<std::vector<cv::Point>> sample_contours){
    double arrayCU[xs * ys];
    double arrayCV[xs * ys];
    // std::cout<<"adaptive vertex"<<std::endl;
    // std::cout<<"contour.size: "<<sample_contours.size()<<std::endl;

    std::clock_t start = std::clock();

    uint us(0);
    double t(0.03);
    // bbs在ShapeFromNormals中被初始化，其中bbs.umax=referenceKeyframe.umax, bbs.umin=referenceKeyframe.umin, bbs.vmax=referenceKeyframe.vmax, bbs.vmin=referenceKeyframe.vmin
    // 576*720大小的画幅
    // double umaxtemp = bbs.umax+0.03; 
    // double umintemp = bbs.umin+0.04; 
    // double vmaxtemp = bbs.vmax-0.04; 
    // double vmintemp = bbs.vmin; 

    double umaxtemp = bbs.umax; 
    double umintemp = bbs.umin; 
    double vmaxtemp = bbs.vmax; 
    double vmintemp = bbs.vmin; 

    // double umaxtemp = bbs.umax-0.065; 
    // double umintemp = bbs.umin+0.015; 
    // double vmaxtemp = bbs.vmax-0.04; 
    // double vmintemp = bbs.vmin+0.05; 
    // std::cout<<"umaxtemp: "<<umaxtemp<<" umintemp: "<<umintemp<<" vmaxtemp: "<<vmaxtemp<<" vmintemp: "<<vmintemp<<std::endl;
    double umax = -1e+6, umin = 1e+6, vmax = -1e+6, vmin = 1e+6;
    for (uint x(0); x < xs; x++)
    {
      for (uint j(0); j < ys; j++)
      {
        arrayCU[us] = double((umaxtemp - umintemp - 2 * t) * x) / (xs - 1) + (umintemp + t);
        arrayCV[us] = double((vmaxtemp - vmintemp - 2 * t) * j) / (ys - 1) + (vmintemp + t);
        // arrayCU[us] = double((umaxtemp - umintemp) * x) / (xs - 1) + (umintemp);
        // arrayCV[us] = double((vmaxtemp - vmintemp) * j) / (ys - 1) + (vmintemp);
        if (umax - arrayCU[us] < 1e-6){
          umax = arrayCU[us];
        }
        if (umin - arrayCU[us] > 1e-6){
          umin = arrayCU[us];
        }
        if (vmax - arrayCV[us] < 1e-6){
          vmax = arrayCV[us];
        }
        if (vmin - arrayCV[us] > 1e-6){
          vmin = arrayCV[us];
        }
        us++;
      }
    }

    // 先形成xs行ys列的网格triangulation
    CDT triangulation;
    for (int i = 0; i < xs*ys; i++){
      triangulation.insert(Point(arrayCU[i], arrayCV[i]));
    }
    cv::Mat triangulation_img = cv::Mat::zeros(height, width, CV_8UC1);
    drawTriangulation(triangulation, triangulation_img, umintemp, umaxtemp, vmintemp, vmaxtemp);
    // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/triangulation_img_"+std::to_string(start)+".jpg", triangulation_img);

    // 向网格triangulation中插入轮廓点
    for (int i = 0; i < sample_contours.size(); i++){
      for (int j = 0; j < sample_contours[i].size(); j++){
        float x = ((float)sample_contours[i][j].x/(float)width) * (umax - umin) + umin;
        float y = ((float)sample_contours[i][j].y/(float)height) * (vmax - vmin) + vmin;
        // Face_handle f = triangulation.locate(Point(x, y));
        // triangulation.insert(Point(x, y), f);
        triangulation.insert(Point(x, y));
      }
    }
    cv::Mat triangulation_insert = cv::Mat::zeros(height, width, CV_8UC1);
    drawTriangulation(triangulation, triangulation_insert, umin, umax, vmin, vmax);
    // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/triangulation_insert_"+std::to_string(start)+".jpg", triangulation_insert);

    // 对网格triangulation世界边缘约束
    for (int i = 1; i < ys; i++){
      double vtemppre = double((vmaxtemp - vmintemp - 2 * t) * (i-1)) / (ys - 1) + (vmintemp + t);
      double vtempnext = double((vmaxtemp - vmintemp - 2 * t) * (i)) / (ys - 1) + (vmintemp + t);
      triangulation.insert_constraint(Point(umin, vtemppre), Point(umin, vtempnext)); // 最左边一列
      triangulation.insert_constraint(Point(umax, vtemppre), Point(umax, vtempnext)); // 最右边一列
    }
    for (int i = 1; i < xs; i++){
      double utemppre = double((umaxtemp - umintemp - 2 * t) * (i-1)) / (xs - 1) + (umintemp + t);
      double utempnext = double((umaxtemp - umintemp - 2 * t) * (i)) / (xs - 1) + (umintemp + t);
      triangulation.insert_constraint(Point(utemppre, vmin), Point(utempnext, vmin)); // 最上边一行
      triangulation.insert_constraint(Point(utemppre, vmax), Point(utempnext, vmax)); // 最下边一行
    }

    // 对网格triangulation进行Lloyd优化
    CGAL::lloyd_optimize_mesh_2(triangulation, CGAL::parameters::max_iteration_number = 2);
    cv::Mat triangulation_lloyd = cv::Mat::zeros(height, width, CV_8UC1);
    drawTriangulation(triangulation, triangulation_lloyd, umin, umax, vmin, vmax);
    // cv::imwrite("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/triangulation_lloyd_"+std::to_string(start)+".jpg", triangulation_lloyd);

    // 为BBS::EvalEigen()准备数据
    // std::ofstream fout("/home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/processInfo/adaptiveVertex.txt");
    int num_of_vertices = triangulation.number_of_vertices();
    Eigen::MatrixXd Val2(num_of_vertices, 1);
    double adaptiveArrayCU[num_of_vertices];
    double adaptiveArrayCV[num_of_vertices];
    int index = 0;
    for (auto vertex = triangulation.finite_vertices_begin(); vertex != triangulation.finite_vertices_end(); vertex++){
      double x = vertex->point().x();
      double y = vertex->point().y();
      adaptiveArrayCU[index] = x;
      adaptiveArrayCV[index] = y;
      vertex->set_sizing_info(index);
      // fout<<"srcU:"<<adaptiveArrayCU[index]<<" srcV:"<<adaptiveArrayCV[index]<<std::endl;
      index++;
    }

    // B-Spline曲面求解
    std::cout<<"number of vertices: "<<num_of_vertices<<std::endl;
    auto start_all = std::chrono::steady_clock::now(); //开始时间
    // std::cout<<"begining adaptive template solve"<<std::endl;
    ofstream BBStime("mapbbs.txt", std::ios::in | std::ios::out | std::ios::ate);
    auto end_all = std::chrono::steady_clock::now(); //开始时间
    auto duration_whole = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
    BBStime<<duration_whole.count()<<std::endl;
    BBStime.close();
    
    BBS::EvalEigen(&bbs, static_cast<const double *>(nodesDepth_), adaptiveArrayCU,
                   adaptiveArrayCV, num_of_vertices, Val2, 0, 0); // 这里相当于解出每个点对应在三维模板空间中的深度Val2
                
    // fout<<"bbs.umin: "<<bbs.umin<<"bbs.umax: "<<bbs.umax<<"bbs.vmin: "<<bbs.vmin<<"bbs.vmax: "<<bbs.vmax<<endl;
    // fout<<"umin: "<<umin<<"umax: "<<umax<<"vmin: "<<vmin<<"vmax: "<<vmax<<endl;
    // 保存adaptive template的结果
    NodesSurface.reserve(num_of_vertices);
    for (uint x(0); x < num_of_vertices; x++)
    {
      cv::Mat x3D(4, 1, CV_32F);
      x3D.at<float>(0, 0) = adaptiveArrayCU[x] * Val2(x, 0);
      x3D.at<float>(1, 0) = adaptiveArrayCV[x] * Val2(x, 0);
      x3D.at<float>(2, 0) = Val2(x, 0);
      x3D.at<float>(3, 0) = 1;
      // std::cout << "x3D: " << x3D << std::endl;
      // fout<<"2D: "<<adaptiveArrayCU[x]<<" "<<adaptiveArrayCV[x]<<endl;
      // fout<<x3D.at<float>(0, 0)<<" "<<x3D.at<float>(1, 0)<<" "<<x3D.at<float>(2, 0)<<std::endl;
      NodesSurface.push_back(x3D);
    }
    // fout.close();

    // 对网格的拓扑信息进行保存（每个面片的三个顶点的索引）
    facets.clear();
    cout<<"number of faces: "<<triangulation.number_of_faces()<<endl;
    for (auto face = triangulation.finite_faces_begin(); face != triangulation.finite_faces_end(); face++){
      Vertex_handle v1 = face->vertex(0);
      Vertex_handle v2 = face->vertex(1);
      Vertex_handle v3 = face->vertex(2);
      std::vector<int> facet;
      facet.push_back(v1->sizing_info());
      facet.push_back(v2->sizing_info());
      facet.push_back(v3->sizing_info());
      facets.push_back(facet);
    }
    // cout<<"end adaptive template solve"<<endl;

    // MyMesh mesh;
    // MyMesh::VertexHandle vhandles[num_of_vertices];
    // for (int i = 0; i < num_of_vertices; i++)
    // {
    //   cv::Mat point = NodesSurface[i];
    //   MyMesh::Point p(point.at<float>(0, 0), point.at<float>(1, 0), point.at<float>(2, 0));
    //   vhandles[i] = mesh.add_vertex(p);
    // }

    // std::vector<MyMesh::VertexHandle> face_vhandles;
    // for (int i = 0; i < facets.size(); i++)
    // {
    //   face_vhandles.clear();
    //   face_vhandles.push_back(vhandles[facets[i][0]]);
    //   face_vhandles.push_back(vhandles[facets[i][1]]);
    //   face_vhandles.push_back(vhandles[facets[i][2]]);
    //   mesh.add_face(face_vhandles);
    // }

    // try
    // {
    //   if (!OpenMesh::IO::write_mesh(mesh, "templateMesh.obj"))
    //   {
    //     std::cerr << "无法写入文件 'output.obj'" << std::endl;
    //   }
    // }
    // catch (std::exception& x)
    // {
    //   std::cerr << x.what() << std::endl;
    // }
    
  }

} // namespace defSLAM
