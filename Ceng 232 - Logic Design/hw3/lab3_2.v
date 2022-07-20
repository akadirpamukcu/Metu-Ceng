`timescale 1ns / 1ps 
module lab3_2(
			input[3:0] command,
			input CLK, 
			input mode,
			output reg [7:0] total_time1,
			output reg [7:0] total_time0,
			output reg [7:0] total_cost1,
			output reg [7:0] total_cost0,
			output reg [3:2] coord_x,
			output reg [1:0] coord_y,
			output reg warning
    );
	 initial begin
		total_time1 <= 8'b00000000;
		total_time0 <= 8'b00000000;
		total_cost1 = 8'b00000000;
		total_cost0 = 8'b00000000;
		coord_x <= 2'b01;
		coord_y <= 2'b01;
		warning <= 1'b0;
	end
	always@(posedge(CLK))
	 begin
	 if(mode == 1)
		begin
			warning<=1;
			total_time1 = 8'b00000000;
			total_time0 = 8'b00000000;
			total_cost1 = 8'b00000000;
			total_cost0 = 8'b00000000;			
			coord_x <= 2'b00;
			coord_y <= 2'b00;		
		end
	else
		begin
		warning <= 0;
		total_time0 = total_time0+1;
			case(command)
				4'b0000: begin
					total_cost0 = total_cost0+1;
					end
				4'b0001: begin
					if(coord_x == 0 || coord_y==2)
						begin
						total_cost0 = total_cost0+1;
						end
					else
						begin
						total_cost0 = total_cost0+3;
						coord_y <= coord_y+1;
						coord_x <= coord_x-1;
						end
					end
				4'b0010: begin
					total_cost0 = total_cost0+2;
					if(coord_y==2)
						begin
						coord_y<= 0;
						end
					else
						begin
						coord_y<= coord_y+1;
						end
					end
				4'b0011: begin
					if(coord_x == 2 || coord_y==2)
						begin
						total_cost0 = total_cost0+1;
						end
					else
						begin
						total_cost0 = total_cost0+3;
						coord_y <= coord_y+1;
						coord_x <= coord_x+1;
						end
					end
				4'b0100: begin
					total_cost0 = total_cost0+2;
					if(coord_x==2)
						begin
						coord_x<= 0;
						end
					else
						begin
						coord_x<= coord_x+1;
						end
					end
				4'b0101: begin
					if(coord_x == 2 || coord_y==0)
						begin
						total_cost0 = total_cost0+1;
						end
					else
						begin
						total_cost0 = total_cost0+3;
						coord_y <= coord_y-1;
						coord_x <= coord_x+1;
						end					
					end
				4'b0110: begin
					total_cost0 = total_cost0+2;
					if(coord_y==0)
						begin
						coord_y<= 2;
						end
					else
						begin
						coord_y<= coord_y-1;
						end				
					end
				4'b0111: begin
					if(coord_x == 0 || coord_y == 0)
						begin
						total_cost0 = total_cost0+1;
						end
					else
						begin
						total_cost0 = total_cost0+3;
						coord_y <= coord_y-1;
						coord_x <= coord_x-1;
							end				
					end
				4'b1000: begin
					total_cost0 = total_cost0+2;
					if(coord_x==0)
						begin
						coord_x<= 2;
						end
					else
						begin
						coord_x<= coord_x-1;
						end
					end				
			endcase
			if(total_cost0 > 9)
			begin
				total_cost0 = total_cost0-10;
				if((total_cost1 + 1) == 2)
					begin
						total_cost1=0;
					end
				else
					begin
						total_cost1= total_cost1+1;
					end
			end
			if(total_time0 > 9)
			begin
				total_time0 = total_time0-10;
				if((total_time1 + 1) == 2)
					begin
						total_time1=0;
					end
				else
					begin
						total_time1= total_time1+1;
					end
			end
			
		
		
		end
	
	
	
	
	 end
	

   //Modify the lines below to implement your design .

endmodule