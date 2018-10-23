#################
#    HLS4ML
#################
open_project -reset encoder_prj
set_top encoder_top
add_files firmware/encoder.cpp -cflags "-I[file normalize ../nnet_utils]"
add_files -tb encoder_test.cpp -cflags "-I[file normalize ../nnet_utils]"
add_files -tb firmware/weights
#add_files -tb tb_data
open_solution -reset "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog
exit
