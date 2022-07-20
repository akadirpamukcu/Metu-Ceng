// Code your design here
`timescale 1ns / 1ps
module PharmacyMem(
	input [4:0] studentID ,
    input clk,
    input [1:0] mode , // 0: list, 2: checkIn, 3: delete
    input [7:0] checkInTime,
    output reg [4:0] listOutput,
    output reg listBusy ,
    output reg ready
	);

integer size;
integer i;
integer j;
integer left;
integer right;
integer smallest;
integer index;
reg [7:0] heap [0:9];
reg [4:0] studentlist[0:9];
reg [4:0] student_temp;
reg [7:0] heap_temp;

initial begin
	ready=0;
	size =0;
	index=0;
	studentlist[0] = 5'b00000;
    studentlist[1] = 5'b00000;
    studentlist[2] = 5'b00000;
    studentlist[3] = 5'b00000;
    studentlist[4] = 5'b00000;
    studentlist[5] = 5'b00000;
    studentlist[6] = 5'b00000;
    studentlist[7] = 5'b00000;
    studentlist[8] = 5'b00000;
    studentlist[9] = 5'b00000;


end 

always@(posedge(clk))
begin
	i=0;
    listBusy = 0;
	case(mode)
		2'b10: begin
		  index=0;
          listBusy = 0;
          if(size <= 9) begin
				size = size + 1;
				i = size-1;
				heap[i] = checkInTime;
				studentlist[i] = studentID;
				for(j=0; j<10; j=j+1)
					begin
					if( i != 0 && (heap[((i-1)/2)] > heap[i]) ) begin
						heap_temp = heap[(i-1)/2];
						heap[(i-1)/2] = heap[i];
						heap[i] = heap_temp;
						student_temp = studentlist[(i-1)/2];
						studentlist[(i-1)/2] = studentlist[i];
						studentlist[i] = student_temp;
						i= (i-1)/2;
					end
				end
			end
		end
		2'b11: begin //delete
			listBusy = 0;
          	index=0;
			if(size == 1) begin
				size=size-1;
			end
			if(size >= 1) begin
				heap[0] = heap[(size-1)];
				studentlist[0] = studentlist[(size-1)];
				size= size -1;
				i=0;
				for(j=0; j<10; j=j+1)
				begin
					left = (2*i)+1;
					right = (2*i)+2;
					smallest = i;
					if( left < size && heap[left] < heap[i]) begin
						smallest = left;
					end
					if( right < size && heap[right] < heap[smallest]) begin
						smallest = right;
					end
					if( smallest != i) begin
					
						heap_temp = heap[i];
						heap[i] = heap[smallest];
						heap[smallest] = heap_temp;
						student_temp = studentlist[i];
						studentlist[i] = studentlist[smallest];
						studentlist[smallest] = student_temp;
						i = smallest;
						
					end
				end
			end
		end
		2'b00: begin // list mode
			
		   ready=0;
          if(index<size) begin
              	listBusy = 1;
				listOutput = studentlist[index];
				index = index +1;
			end
          else if(index>=size) begin
				ready=1;
				index=0;
			end
			
		
		end
		
	endcase
end

endmodule