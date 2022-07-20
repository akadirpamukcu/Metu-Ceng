

module hasher (
	input clk,    // Clock
	input [15:0] cur_time,
	input [15:0] student_id,
	output [15:0] cur_hash	
);
  reg [15:0] cur_hashlama;

initial begin
	cur_hashlama = 0;
end
  
  
always@(posedge clk) begin
  cur_hashlama = ( ( ( (cur_time ^ student_id) - cur_hashlama) *((cur_time ^ student_id) - cur_hashlama) ) >> 8) & 65535 ;

  
end

assign cur_hash = cur_hashlama; 
  
endmodule