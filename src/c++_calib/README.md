mkdir build && cd build
cmake ..
make


./calibrate -w [board_width] -h [board_height] -n [num_imgs] -s [square_size] -d [imgs_directory] -i [imgs_filename] -e [file_extension] -o [output_filename]

./calibrate_stereo -n [num_imgs] -u [left_cam_calib] -v [right_cam_calib] -L [left_img_dir] -R [right_img_dir] -l [left_img_prefix] -r [right_img_prefix] -o [output_calib_file] -e [file_extension]

./undistort_rectify -l [left_img_path] -r [right_img_path] -c [stereo_calib_file] -L [output_left_img] -R [output_right_img]


./calibrate --board_width 8 --board_height 6 --num_imgs 35 --square_size 0.0236 --imgs_directory ../../../images/left/ --imgs_filename "" --extension jpg --out_file ../../../calib_files/cam_left.yml

./calibrate --board_width 8 --board_height 6 --num_imgs 35 --square_size 0.0236 --imgs_directory ../../../images/right/ --imgs_filename "" --extension jpg --out_file ../../../calib_files/cam_right.yml

./calibrate_stereo --num_imgs 35 --leftcalib_file ../../../calib_files/cam_left.yml --rightcalib_file ../../../calib_files/cam_right.yml --leftimg_dir ../../../images/left/ --rightimg_dir ../../../images/right/ --leftimg_filename "" --rightimg_filename "" --extension jpg --out_file ../../../calib_files/stereo_cam.yml