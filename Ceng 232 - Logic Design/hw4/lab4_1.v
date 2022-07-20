`timescale 1ns / 1ps

module AddParity(
input [1:8] dataIn,
output reg [1:12] dataOut
);
initial 
	begin
		dataOut = 12'b000000000000;
	end

always@(dataIn)
	begin
		dataOut[1] = dataIn[1] ^ dataIn[2] ^ dataIn[4] ^ dataIn[5] ^ dataIn[7];
		dataOut[2] = dataIn[1] ^ dataIn[3] ^ dataIn[4] ^ dataIn[6] ^ dataIn[7];
		dataOut[3] = dataIn[1];
		dataOut[4] = dataIn[2] ^ dataIn[3] ^ dataIn[4] ^ dataIn[8];
		dataOut[5] = dataIn[2];
		dataOut[6] = dataIn[3];
		dataOut[7] = dataIn[4];
		dataOut[8] = dataIn[5] ^ dataIn[6] ^ dataIn[7] ^ dataIn[8];
		dataOut[9] = dataIn[5];
		dataOut[10] = dataIn[6];
		dataOut[11] = dataIn[7];
		dataOut[12] = dataIn[8];
	end
endmodule

module CheckParity(
input [1:12] dataIn,
output reg [1:8] dataOut
);


reg [1:4] parity;
always@(*)
	begin
		dataOut = {dataIn[3], dataIn[5], dataIn[6], dataIn[7], dataIn[9], dataIn[10], dataIn[11],dataIn[12]};
		parity[4] = dataIn[1] ^ dataIn[3] ^ dataIn[5] ^ dataIn[7] ^ dataIn[9] ^ dataIn[11] ;
		parity[3]= dataIn[2] ^ dataIn[3] ^ dataIn[6] ^ dataIn[7] ^ dataIn[10] ^ dataIn[11];
		parity[2]= dataIn[4] ^ dataIn[5] ^ dataIn[6] ^ dataIn[7] ^ dataIn[12];
		parity[1]= dataIn[8] ^ dataIn[9] ^ dataIn[10] ^ dataIn[11] ^ dataIn[12];
		case(parity)
			4'b0011: begin //3
				dataOut[1] = ~dataOut[1];
			end
			
			4'b0101: begin //5
				dataOut[2] = ~dataOut[2];
			end
			
			4'b0110: begin //6
				dataOut[3] = ~dataOut[3];				
			end
			
			4'b0111: begin //7
				dataOut[4] = ~dataOut[4];	 
			end
			
			4'b1001: begin //9
				dataOut[5] = ~dataOut[5];	
			end
			
			4'b1010: begin //10
				dataOut[6] = ~dataOut[6];	
			end
			
			4'b1011: begin //11
				dataOut[7] = ~dataOut[7];	
			end
			
			4'b1100: begin //12
				dataOut[8] = ~dataOut[8];	
			end
		endcase
	end

//Write your code below
//
//
endmodule


module RAM(
input [7:0] dataIn, //0:read, 1:write
input clk,
input mode,
input [3:0] addr,
output reg [7:0] dataOut);

//Write your code below
//
//
reg [7:0] ram[15:0];
integer i;
initial begin
	for(i=0; i<16; i=i+1)
		begin
			ram[i] = 0;
		end
end

always@(posedge(clk))
begin
	if(mode == 1)
		begin
			ram[addr] <= dataIn;			
		end
	if(mode == 0)
		begin
			dataOut <= ram[addr];
		end
end
endmodule

module SEC_MEM(
input [1:12] dataIn,
input clk,
input mode, //0:read, 1:write
input [3:0] addr,
output  [1:12] dataOut);

//DO NOT EDIT THIS MODULE
wire  [7:0]  ramDataIn;
wire  [7:0]  ramDataOut;
CheckParity CP(dataIn,ramDataIn);
RAM RM(ramDataIn, clk, mode, addr, ramDataOut);
AddParity AP(ramDataOut,dataOut);

endmodule
