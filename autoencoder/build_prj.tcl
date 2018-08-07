#################
#    HLS4ML
#################
open_project -reset autoencoder_prj
set_top autoencoder
add_files firmware/autoencoder.cpp -cflags "-I[file normalize ../nnet_utils]"
add_files -tb autoencoder_test.cpp -cflags "-I[file normalize ../nnet_utils]"
add_files -tb firmware/weights
#add_files -tb tb_data
open_solution -reset "solution1"
set_part {xc7vx690tffg1927-2}
create_clock -period 5 -name default
csim_design
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog
exit
