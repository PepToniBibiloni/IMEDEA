# Script para hacer la calibración

# ../src/c++_calib/build/calibrate --board_width 8 --board_height 6 --num_imgs [Num imagenes para calibrar] --square_size [Tamaño cuadrado tablero] --imgs_directory [Ubicación imagenes] --imgs_filename [Palabra anterior al numero] --extension [Extension de las imagenes] --out_file [Nombre fitxero resultante]
../src/c++_calib/build/calibrate --board_width 8 --board_height 6 --num_imgs 52 --square_size 0.0236 --imgs_directory ../data/left/ --imgs_filename "" --extension jpg --out_file cam_left.yml

../src/c++_calib/build/calibrate --board_width 8 --board_height 6 --num_imgs 52 --square_size 0.0236 --imgs_directory ../data/right/ --imgs_filename "" --extension jpg --out_file cam_right.yml

../src/c++_calib/build/calibrate_stereo --num_imgs 35 --leftcalib_file cam_left.yml --rightcalib_file cam_right.yml --leftimg_dir ../data/both/ --rightimg_dir ../data/both/ --leftimg_filename "left" --rightimg_filename "right" --extension jpg --out_file stereo_cam.yml


