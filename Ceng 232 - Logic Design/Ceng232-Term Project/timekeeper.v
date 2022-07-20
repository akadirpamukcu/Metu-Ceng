

module timekeeper (
	input clk,    // Clock
	output[15:0] cur_time	
);


reg [15:0] timey;
  
initial begin
  timey=0;
end
  
always@(posedge clk) begin
  if(timey == 16'b1111111111111111) begin
    timey<=0;
  end
  	
  	timey<= timey +1;
end
  
assign cur_time = timey;

  
  
  
endmodule
