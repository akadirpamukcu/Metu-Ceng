

module b16toBCD (
	input [15:0] to_display,
	input enable,
	output  [3:0] D5,	
	output  [3:0] D4,	
	output  [3:0] D3,	
	output  [3:0] D2,	
	output  [3:0] D1	
);
  	reg [3:0] d5;
  	reg [3:0] d4;
  	reg [3:0] d3;
  	reg [3:0] d2;
  	reg [3:0] d1;
  	
  	reg [0:4] i;
  	reg [0:4] j;
    reg [20:0] bcd;
  	
  always @(*) begin
    if(enable == 0) begin
   	  for(i = 0; i <= 20; i = i+1) begin
        	bcd[i] = 1;                        
        end
    end
    else
      begin
        for(i = 0; i <= 20; i = i+1) begin
            bcd[i] = 0;                        
        end
        
        bcd[15:0] = to_display;
        for(i=0; i<= 12; i= i+1) begin
          for(j=0; j<= i/3; j = j+1) begin
            if( bcd[16-i+4*j -: 4] >4) begin
              bcd[16-i+4*j -: 4] = bcd[16-i+4*j -: 4] + 4'd3;
            end
          end
        end
        
      end
      {d5,d4,d3,d2,d1} = bcd;
    end
    assign D5 = d5;
    assign D4 = d4;
    assign D3 = d3;
    assign D2 = d2;
    assign D1 = d1;
      
          
endmodule