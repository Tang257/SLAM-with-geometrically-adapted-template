# Run camera
#./DefSLAM /home/jose/DeformableSLAM/Vocabulary/ORBvoc.txt /home/jose/DeformableSLAM/calibration_files/logitechc922.yaml
# Run one video without ground truth
#./DefSLAM /home/jose/DeformableSLAM/Vocabulary/ORBvoc.txt /media/jose/NuevoVol/videosDataset/sequence_heart/hamlyn.yaml /media/jose/NuevoVol/videosDataset/f5phantom/f5_dynamic_deint_L.avi

# black BK
./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/phantom_datasets/camera.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/phantom_datasets/video/Capture_20231121_Black_BK/Liver_BK_tools.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/camera_contour.yaml /home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/video/Capture_20231020_1.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/camera_contour.yaml /home/tang/GithubRep/SD-DefSLAM/datasets/phantom_datasets/video/Capture_20230921_5.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/SD-DefSLAM_contour_AMMNRR/datasets/phantom_datasets/camera.yaml /home/tang/Github/SD-DefSLAM_contour_AMMNRR/datasets/phantom_datasets/video/Capture_20231020_1.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/SD-DefSLAM_contour_AMMNRR/datasets/phantom_datasets/camera.yaml /home/tang/Github/SD-DefSLAM_contour_AMMNRR/datasets/phantom_datasets/video/Capture_20230921_5.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/GithubRep/SD-DefSLAM/datasets/jinyu/process_datasets/camera/cut2_short2.yaml /home/tang/GithubRep/SD-DefSLAM/datasets/jinyu/process_datasets/explore_liver_1min.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/phantom_datasets/camera.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/phantom_datasets/video/Capture_20231019/Capture_20231019-normal-phase.avi

./DefSLAM /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/phantom_datasets/camera.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/phantom_datasets/video/Capture_20231019_tools/Capture_20231019_tools-phase1.avi

# Groundtruth depth image
./DefSLAMGTCT /home/jose/DefKLTSLAM/Vocabulary/ORBvoc.txt /media/jose/NuevoVol/videosDataset/f5phantom/hamlyn.yaml /media/jose/NuevoVol/videosDataset/f5phantom/f5_dynamic_deint_L.avi /media/jose/NuevoVol/videosDataset/f5phantom/f5/heartDepthMap_

# Groundtruth stereo
#./DefSLAMGT /home/jose/DefKLTSLAM/Vocabulary/ORBvoc.txt /media/jose/NuevoVol/videosDataset/Jose/stereo3.yaml /media/jose/NuevoVol/videosDataset/Jose/Mandala3/images /media/jose/NuevoVol/videosDataset/Jose/Mandala3/images /media/jose/NuevoVol/videosDataset/Jose/Mandala3/timestamps/timestamps.txt

#./DefSLAMHamyln /home/jose/DeformableSLAM/Vocabulary/ORBvoc.txt /media/jose/NuevoVol/videosDataset/Jose/stereo3.yaml /media/jose/NuevoVol/videosDataset/Jose/Mandala3/images /media/jose/NuevoVol/videosDataset/Jose/Mandala3/images /media/jose/NuevoVol/videosDataset/Jose/Mandala3/timestamps/timestamps.txt

# mandala0
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/stereo_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala0/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala0/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala0/timestamps/timestamps.txt

# mandala1
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/stereo_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala1/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala1/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala1/timestamps/timestamps.txt

# mandala2
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/stereo_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala2/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala2/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala2/timestamps/timestamps.txt

# mandala3
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/stereo_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala3/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala3/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala3/timestamps/timestamps.txt

# mandala4
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/stereo_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala4/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala4/images /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala4/timestamps/timestamps.txt

./DefSLAM /home/tang/Github/SD-DefSLAM/Vocabulary/ORBvoc.txt /home/tang/GithubRep/SD-DefSLAM/datasets/mandala/stereo_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/MandalaDataset/Mandala3/mandala3.avi

# f5phantom
./DefSLAMGTCT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/f5phantom/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/f5phantom/f5_dynamic_deint_L.avi /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/f5phantom/f5/heartDepthMap_

# f7phantom
./DefSLAMGTCT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/f7phantom/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/f7phantom/f7_dynamic_deint_L.avi /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/f7phantom/f7/heartDepthMap_

# sequence_abdomen
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_abdomen/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_abdomen/camara0 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_abdomen/camara1 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_abdomen/times.txt

# sequence_exploration
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_exploration/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_exploration/camara0 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_exploration/camara1 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_exploration/times.txt

# sequence_heart
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_heart/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_heart/camara0 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_heart/camara1 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_heart/times.txt

# sequence_organs
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_organs/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_organs/camara0 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_organs/camara1 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_organs/times.txt

# sequence_tool
./DefSLAMGT /home/tang/Github/SD-DefSLAM_contour/Vocabulary/ORBvoc.txt /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_tool/hamlyn_contour.yaml /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_tool/camara0 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_tool/camara1 /home/tang/Github/DefSLAM/ExperimentDatabase/HamlynDatasetShort/sequence_tool/timestamp.txt
