#################
#    HLS4ML
#################
open_project -reset decoder_prj
set_top decoder_top
add_files firmware/decoder.cpp -cflags "-I[file normalize ../nnet_utils]"
add_files -tb decoder_test.cpp -cflags "-I[file normalize ../nnet_utils]"
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
