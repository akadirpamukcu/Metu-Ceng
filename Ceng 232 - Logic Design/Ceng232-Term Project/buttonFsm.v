

module buttonFsm (
	input clk,
	input button,    
	output stateful_button
);
reg old_value;
reg stateful;

initial begin
  stateful=0;
  old_value=0;
end
  

always@(posedge clk) begin
  stateful = old_value;
end


  always@(posedge button)
	begin
      if(stateful == 0 && old_value == 0) begin
      	stateful = 1;
        old_value=1;
      end
      else if(stateful== 0 && old_value == 1 ) begin
        stateful=1;      	
      end
      else if(stateful ==1 && old_value==1) begin
      	stateful=0;
        old_value=0;
      end
      else if(stateful==1 && old_value==0) begin
      	stateful=1;
      end
      
    end

  
assign stateful_button = stateful;
  
endmodule
