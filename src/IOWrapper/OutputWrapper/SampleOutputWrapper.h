/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include <fstream>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "util/NumType.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class SampleOutputWrapper : public Output3DWrapper
{
public:
        inline SampleOutputWrapper(std::string depth_dir, std::string pose, std::string keyframe, std::string pose_tum, std::string keyframe_tum)
        {
            pose_out.open(pose.c_str());
            final_kf_poses_out.open(keyframe.c_str());
            pose_out_tum.open(pose_tum.c_str());
            final_kf_poses_out_tum.open(keyframe_tum.c_str());
            depthout_dir = depth_dir.c_str();
            printf("OUT: Created SampleOutputWrapper\n");
        }

        virtual ~SampleOutputWrapper()
        {
            pose_out.close();
            final_kf_poses_out.close();
            pose_out_tum.close();
            final_kf_poses_out_tum.close();
            printf("OUT: Destroyed SampleOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
            return;
            printf("OUT: got graph with %d edges\n", (int)connectivity.size());

            int maxWrite = 5;

            for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
            {
                int idHost = p.first>>32;
                int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
                printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
                maxWrite--;
                if(maxWrite==0) break;
            }
        }



        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool finalize, CalibHessian* HCalib) override
        {
            if(finalize){
                for(FrameHessian* f : frames){
                    // output frames
                    Eigen::Matrix<double, 3, 4> pose = f->shell->camToWorld.matrix3x4();
                    for(int i = 0; i < 3; i++){
                        for(int j = 0; j < 4; j++){
                            final_kf_poses_out << pose(i, j) << " ";
                        }
                    }
                    final_kf_poses_out << f->shell->incoming_id << "\n";
                    final_kf_poses_out.flush();

                    final_kf_poses_out_tum << f->shell->incoming_id <<
			        " " << f->shell->camToWorld.translation().transpose().x()<<
                    " " << f->shell->camToWorld.translation().transpose().y()<<
                    " " << f->shell->camToWorld.translation().transpose().z()<<
			        " " << f->shell->camToWorld.so3().unit_quaternion().x()<<
			        " " << f->shell->camToWorld.so3().unit_quaternion().y()<<
			        " " << f->shell->camToWorld.so3().unit_quaternion().z()<<
			        " " << f->shell->camToWorld.so3().unit_quaternion().w() << "\n";

                    final_kf_poses_out_tum.flush();
                    // output sparse depth values

                    // printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
                    //     f->frameID,
                    //     finalize ? "final" : "non-final",
                    //     f->shell->incoming_id,
                    //     f->shell->timestamp,
                    //     (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
                    // std::cout << f->shell->camToWorld.matrix3x4() << "\n";

                    char depthout_name_buf[20];
                    sprintf(depthout_name_buf, "%06d.bin", f->shell->incoming_id);
                    std::string depthout_path = depthout_dir + depthout_name_buf;
                    std::ofstream depthout;
                    depthout.open(depthout_path.c_str(), std::ios::binary | std::ios::out);
                    int num_points = f->pointHessiansMarginalized.size() + f->pointHessians.size();

                    std::vector<PointHessian*> points = f->pointHessiansMarginalized;
                    points.insert(points.end(), f->pointHessians.begin(), f->pointHessians.end());

                    if(frames.size() == 1)
                        assert(f->pointHessians.size() == 0);

                    depthout.write(reinterpret_cast<char*>(&num_points), sizeof(num_points));
                    for(PointHessian* p : points){
                        float idepth = p->idepth;
                        // float depth4 = depth*depth; depth4 *= depth4;
                        float idepth_hessian = p->idepth_hessian;
                        float maxRelBaseline = p->maxRelBaseline;
                        int numGoodResiduals = p->numGoodResiduals;
                        // float var = (1.0f / (p->idepth_hessian+0.01));
                        float u = p->u;
                        float v = p->v;
                        float idepth_scaled = p->idepth_scaled;
                        depthout.write(reinterpret_cast<char*>(&u), sizeof(u));
                        depthout.write(reinterpret_cast<char*>(&v), sizeof(v));
                        depthout.write(reinterpret_cast<char*>(&idepth), sizeof(idepth));
                        depthout.write(reinterpret_cast<char*>(&idepth_scaled), sizeof(idepth_scaled));
                        depthout.write(reinterpret_cast<char*>(&idepth_hessian), sizeof(idepth_hessian));
                        depthout.write(reinterpret_cast<char*>(&maxRelBaseline), sizeof(maxRelBaseline));
                        depthout.write(reinterpret_cast<char*>(&numGoodResiduals), sizeof(numGoodResiduals)); //int
                    }
                    depthout.close();
                }
            }
        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
            // printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
            //        frame->incoming_id,
            //        frame->timestamp,
            //        frame->id);
            Eigen::Matrix<double, 3, 4> pose = frame->camToWorld.matrix3x4();
            // std::cout << frame->camToWorld.matrix3x4() << "\n";
            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 4; j++){
                    pose_out << pose(i, j) << " ";
                }
            }
            pose_out << frame->incoming_id << "\n";
            pose_out.flush();
            pose_out_tum << frame->incoming_id <<
            " " << frame->camToWorld.translation().transpose().x()<<
            " " << frame->camToWorld.translation().transpose().y()<<
            " " << frame->camToWorld.translation().transpose().z()<<
            " " << frame->camToWorld.so3().unit_quaternion().x()<<
            " " << frame->camToWorld.so3().unit_quaternion().y()<<
            " " << frame->camToWorld.so3().unit_quaternion().z()<<
            " " << frame->camToWorld.so3().unit_quaternion().w() << "\n";

            pose_out_tum.flush();
        }


        virtual void pushLiveFrame(FrameHessian* image) override
        {
            // can be used to get the raw image / intensity pyramid.
        }

        virtual void pushDepthImage(MinimalImageB3* image) override
        {
            // can be used to get the raw image with depth overlay.
        }
        virtual bool needPushDepthImage() override
        {
            return false;
        }

        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override
        {
            return;
            printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
                   KF->frameID,
                   KF->shell->incoming_id,
                   KF->shell->timestamp,
                   KF->shell->id);
            std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

            int maxWrite = 5;
            for(int y=0;y<image->h;y++)
            {
                for(int x=0;x<image->w;x++)
                {
                    if(image->at(x,y) <= 0) continue;

                    printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
                    maxWrite--;
                    if(maxWrite==0) break;
                }
                if(maxWrite==0) break;
            }
        }
private:
std::ofstream pose_out;
std::ofstream pose_out_tum;
std::ofstream final_kf_poses_out;
std::ofstream final_kf_poses_out_tum;
std::string depthout_dir;
};



}



}
