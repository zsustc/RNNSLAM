Run RNNSLAM (DSO+RNN):
./bin/dso_dataset mode=2 preset=0 files=~/Datasets/COLON_RNNSLAM_TEST/sequences/031/image calib=~/Datasets/COLON_RNNSLAM_TEST/sequences/calib_270_216.txt rnn=~/software/dso/src/RNN rnnmodel=~/Datasets/COLON_RNNSLAM_TEST/models/model-145000 numRNNBootstrap=9 lostTolerance=5 output_prefix=/home/ruibinma/Datasets/COLON_RNNSLAM_TEST/sequences/031/ quiet=1 sampleoutput=1

Run depth map refining
python my_system.py --cameras_file_path ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/kf_pose_result.txt --depth_dir ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/depth --image_dir ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/image/ --intrinsic ~/Datasets/COLON_RNNSLAM_TEST/sequences/calib_270_216.txt --presort_poses --refined_depth_dir ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/depth_refined_145000 --local_window_size 7  --use_viewer

Convert dso output to tumrgbd format
python cvt_colon_to_tumrgbd.py --depth_dir ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/depth_refined_145000/ --image_dir ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/image/ --cameras_file_path ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/kf_pose_result.txt --output_dir ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/tum_refined_145000 --intrinsic ~/Datasets/COLON_RNNSLAM_TEST/sequences/calib_270_216.txt --repeat 1 --high_intensity_threshold 250 --low_intensity_threshold 70 --rescale_w 320 --rescale_h 256

Run surfelmeshing
./SurfelMeshing ~/Datasets/COLON_RNNSLAM_TEST/sequences/031/tum_refined_145000/ trajectory.txt --follow_input_camera false --depth_valid_region_radius 160 --export_mesh mesh_031.obj --outlier_filtering_frame_count 2 --outlier_filtering_required_inliers 1 --observation_angle_threshold_deg 90 --sensor_noise_factor 0.3 --create_video --hide_camera_frustum --render_window_default_width 640 --render_window_default_height 640 --max_depth 2.5 --bilateral_filter_radius_factor 5