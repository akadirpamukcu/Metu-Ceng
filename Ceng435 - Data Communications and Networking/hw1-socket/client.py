import sys, socket
import time, os, hashlib
BUFFER_SIZE = 900 #Byte size of each fragment(chunk) of file.
DELAY = 15 #Client wait for 15 seconds to answer from server and shut down.

def udp_client(host,port,uport): #udp_client function
    max_timeout=0
    sent_again_count=0
    chunks = []
    ack_list = [] #Acknowlegment list, any packet get ACK added here.
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.bind(("",uport)) # Binding udp port to client
    client.settimeout(1) #Timeout before a new packet send to the server is 1 second.
    with open("transfer_file_UDP.txt", "rb") as f: #This section for reading the file completly and written into a list.
        while(True):   
            bytes_read = f.read(BUFFER_SIZE) 
            if not bytes_read:
                break
            chunks.append(bytes_read)
    while(len(ack_list) != len(chunks)): #While loop with condition that every packet sended gets ACK
        if(max_timeout>=DELAY):  #if server not responding for 15 minutes,if true then kill it.
            break
        for i in range(len(chunks)):
            if i not in ack_list:
                header = b'<HS*>' + str(i).encode() + b'<HE*>' #Adding token of header that contains no of this packet
                time_sent = b'<TS*>'+ ((str(time.time())).encode()) +  b'<TE*>' #adding token of sent_time
                packet = (chunks[i]+header+time_sent) #collecting the packet
                checksum = b'<CS*>'+ hashlib.md5(packet).digest() + b'<CE*>' # adding token of md5 checksum 
                packet+=checksum #adding checksum to the packet
                if i == len(chunks)-1:
                    packet+= b'<END*>' #adding an <END> packet to last packet. So that serve can understand.
                client.sendto(packet,(host,port))
                
                try:
                    data,server = client.recvfrom(120) #Listening server for ACK.
                    max_timeout=0 #a packet arrived so server is still working, make timer 0
                    data_finish_index=data.find(b'<CS>') #finding checksum of receving message and checking it.
                    csum_st_index=data_finish_index+4
                    csum_fn_index=data.find(b'<CE>')
                    if( hashlib.md5(data[:data_finish_index]).digest() != data[csum_st_index:csum_fn_index]):
                        sent_again_count+=1 #if checksum of coming message is wrong, send it again
                        continue
                    data = data[:data_finish_index]
                    if(data[:3] == b'ACK'):
                        
                        try: 
                            index = int(data[3:].decode()) #After receiving ACK looking for which packet has ACKed.
                            if index not in  ack_list:
                                ack_list.append(index)
                        except ValueError: #ACK message index corrupted
                            sent_again_count+=1 #incrementing count of packets that sent again
                            continue
                    elif(data[:4] == b'NACK'): #If a packet get NACK try again
                        sent_again_count+=1 #incrementing count of packets that sent again
                        continue
                except socket.timeout: #if socket get timeout error and try to send same packet again.
                    sent_again_count+=1 #incrementing count of packets that sent again
                    max_timeout+=1 #a count for understanding is server down, if it is 15. meaning that server not responding for 15 seconds. then close the client
                    continue
    print("UDP Transmission Re-transferred Packets:", sent_again_count) #print the result.
    client.close() #close the client after we done.





def tcp_client(host,port,cport): #tcp_client function
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.bind(("",cport)) #binding client to tcp port
    client.connect((host,port))
    flag=True
    chunks = []
    with open("transfer_file_TCP.txt", "rb") as f: #This section for reading the file completly and written into a list.
        while(True):   
            bytes_read = f.read(500) 
            if not bytes_read:
                break
            chunks.append(bytes_read)
    i=0
    while(flag):
        time_sent = b'<TS*>'+ ((str(time.time())).encode()) +  b'<TE*>' #adding token of sent_time
        packet = chunks[i] + time_sent
        if i == len(chunks)-1:
            packet+= b'<*END*>'
            flag=False
        client.sendall((packet))
        data = client.recv(50).decode() #waiting for ACK
        if(data == "ACK"):
            i+=1
    client.close() #closing client after we are done.



if __name__ == "__main__": #main function that paramters passes through functions.
    server_ip = (sys.argv[1])
    server_udp_port = int(sys.argv[2])
    server_tcp_port = int(sys.argv[3])
    client_udp_port = int(sys.argv[4])
    client_tcp_port = int(sys.argv[5])
    udp_client(server_ip,server_udp_port,client_udp_port)
    tcp_client(server_ip, server_tcp_port,client_tcp_port)
    
