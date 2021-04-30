import sys, socket
import time, os, hashlib, threading
BUFFER_SIZE = 1000 #Byte size of each packet going to be read.

ACK = "ACK"
NACK = "NACK"
nack_message = (b'NACK') #preapering a NACK message and calculating it's checksum to send every time needed.
nack_message = nack_message + b'<CS>' + hashlib.md5(nack_message).digest() + b'<CE>'

def udp_server(host,port): #udp_server function
    total_packet_count=-1
    packet_count=0
    total_time=0.0
    first_time=0
    full_time=0
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #establishing the socket
    server.bind((host,port))
    packets=[]
    flag=True
    while(True): #running a infinite loop until we got every packet
        data,addr = server.recvfrom(BUFFER_SIZE) #reading 1000 bytes each time.
        recv_time = time.time() #measuring time when the packet arrived
        data_finish_index = data.find(b'<HS*>')  #Finding the locations of informations from tokens.
        header_start_index = data_finish_index+5
        header_finish_index= data.find(b'<HE*>')
        header = data[header_start_index:header_finish_index]
        time_start_index = data.find(b'<TS*>')+5
        time_end_index = data.find(b'<TE*>')
        csum_start_index = data.find(b'<CS*>')+5
        csum_end_index = data.find(b'<CE*>')
        
        if(hashlib.md5(data[:csum_start_index-5]).digest() == data[csum_start_index:csum_end_index]):  #comparing checksums in order to check packet corruption
            try: #trying calculate seq number of packet and sent_time
                header_index = int(header.decode()) 
                time_sent = float((data[time_start_index:time_end_index]).decode())
            except ValueError:
                server.sendto(nack_message, addr) #if the seq number of time corrupted send a NACK
                continue
            ack_message = b'ACK' +header #prepare a NACK message and checksum with packet seq number.
            ack_message = ack_message + b'<CS>' +hashlib.md5(ack_message).digest() +b'<CE>' 
            server.sendto(ack_message, addr)
            packet =(header_index, data[:data_finish_index]) #make every packet is a tuple with content and number
            if packet not in packets:
                packets.append(packet) #adding every packet to a tuple list according to their locations in file (coming from header file)
                total_time += (recv_time - time_sent) #calculating the time of every successfull and adding to the total_time
                packet_count+=1 #counting the every packet that arrive correctly for the first time.
                if flag==True:
                    first_time=(recv_time - time_sent)
                    flag=False

        else:
            server.sendto(nack_message, addr)  #if checksum didnt match send NACK to the client
        if data.find(b'<END*>') != -1:
            total_packet_count=header_index+1 #find the total packet count with checking END token in the last packet.
        if total_packet_count == len(packets): #check if all packets sent if true finish the receiving process
            full_time=recv_time-first_time
            break
    with open("yeni.txt", "wb") as fp:
        for packet in sorted(packets):
            fp.write(packet[1])
    print("UDP Packets Average Transmission Time:",total_time/packet_count  ,"ms" ) #printing results.
    print("UDP Communication Total Transmission Time:",full_time, "ms")
    server.close()

def tcp_server(host,port): #tcp server function
    packet_count=0
    total_time=0
    full_time=0
    first_time=0
    flag =True
    file = bytes() #creating a file bytes objects to write and collect all packets
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #establish TCP connection
    server.bind((host,port))
    server.listen()
    conn, addr = server.accept() 
    while(True):
        data = conn.recv(1000) #taking packets up to 1000 bytes at a time
        rec_time = time.time() #keeping the received time.
        time_start_index = data.find(b'<TS*>')+5 #using token to locate sent_time
        time_end_index = data.find(b'<TE*>')
        sent_time = float((data[time_start_index:time_end_index].decode()))
        if flag==True:
            first_time=sent_time
            flag=False
        file+=data[:time_start_index-5]
        total_time+= rec_time-sent_time  #calculating total_time
        packet_count+=1
        conn.sendall(ACK.encode())
        if(data.find(b'<*END*>') != -1 ):
            full_time=rec_time-first_time
            break
    conn.close()
    with open("transfer_file_TCP.txt", "wb") as fp: #writing packets to a file
        fp.write(file)
    print("TCP Packets Average Transmission Time:",total_time/packet_count  ,"ms" ) #printing results
    print("TCP Communication Total Transmission Time:",full_time, "ms")


if __name__ == "__main__":
    port_udp = int(sys.argv[1]) #passing arguments 
    port_tcp = int(sys.argv[2])
    host  = socket.gethostbyname(socket.gethostname()) #using threadding to make both servers work at the same time.
    thread1= threading.Thread(target=tcp_server, args=(host,port_tcp)) #tcp thread
    thread2= threading.Thread(target=udp_server, args=(host,port_udp)) #udp thread
    thread1.start()
    thread2.start()
