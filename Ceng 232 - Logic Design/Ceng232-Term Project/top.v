module top (
	input sysclk,    // 1 MHz
	input button_in,
	input [15:0] student_id,
	output[3:0] D5_out,	
	output[3:0] D4_out,	
	output[3:0] D3_out,	
	output[3:0] D2_out,	
	output[3:0] D1_out		
);


wire clock500hz;
wire clockN;
wire [15:0] cur_time;
wire [15:0] curr;
wire [15:0] cur_hash;
wire stateful_button;

rtcClkDivider clk_module(sysclk,clock500hz,clockN);
timekeeper time_module(clockN,cur_time);
// assign curr= cur_time+1;
 hasher hash_module(clockN,cur_time,student_id,cur_hash);
buttonFsm button_module(clock500hz,button_in, stateful_button);
b16toBCD b16_module(cur_hash,stateful_button,D5_out,D4_out,D3_out,D2_out,D1_out);



endmodule
