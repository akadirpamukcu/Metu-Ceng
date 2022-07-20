module rtcClkDivider (
	input sys_clk,    // 1 MHz
	output clk_500Hz, // 500 Hz
	output clk_5s  // 0.2 Hz
);
  reg clk_1;
  reg clk_2;
  integer count1=1;
  integer count2=1;
  initial begin
  clk_1 = 0;
  clk_2=0;
end
parameter KEYCHANGE_PERIOD = 5;  // DO NOT CHANGE THE NAME OF THIS PARAMETER 
// AND MAKE SURE TO USE THIS PARAMETER INSTEAD OF HARDCODING OTHER VALUES

  
  always@(posedge sys_clk) begin
    if(count1 == 1000) begin
      	clk_1= ~clk_1;
      	count1=0;
    end
    if(count2 == (KEYCHANGE_PERIOD*500000)) begin
    	count2=0;
      	clk_2=~clk_2;
    end
    count1 = count1+1;
    count2 = count2+1;
    
  end
  assign clk_500Hz = clk_1;
  assign clk_5s = clk_2;
endmodule
