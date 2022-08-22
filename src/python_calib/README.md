python single_camera_calibration.py --image_dir ../images/right/ --image_format jpg --prefix "" --square_size 0.0236 --width 8 --height 6 --save_file right_cam.yml


python stereo_camera_calibration.py --left_file left_cam.yml --right_file right_cam.yml --left_prefix "" --right_prefix "" --left_dir ../images/left/ --right_dir ../images/right/ --image_format jpg --width 8 --height 6 --square_size 0.0236 --save_file stereo_cam.yml

pyhon stereo_depth.py --calibration_file stereo_cam.yml --left_source ../images/left/10.jpg --right_source ../images/right/10.jpg 