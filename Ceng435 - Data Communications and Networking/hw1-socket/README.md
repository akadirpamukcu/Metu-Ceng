Ceng435 THE2 Socket Programming Homework

A simple server and client uses tcp and udp to communicate.


1-), client.py and server.py, written in python3

client.py takes 5 arguments: 

the “server”s IP address,
the “server”s UDP listen port,
the “server”s TCP listen port,
the “client”s sender port for UDP communication,
the “client”s sender port for TCP communication

Ex: python3 client.py 127.0.1.1 1205 1206 1201 1202

server.py take 2 arguments.

the “server”s UDP listen port,
the “server”s TCP listen port,

Client.py has a delay time as 15 seconds, if server does not responds in 15 seconds, it will gonna shut itsef. It can be change. Also server fragments files into 900 bytes pieces in the client side and it receive 1000 bytes in the server side. It can be changed but difference must be preserved. For example if high bandwidths avaliable but file is too big, the BUFFER_SIZE can be very high numbers, for example 10000 or 100000
Vice versa if bandwidht is the problematic side, buffer can be shrinked.

3-) A spesific RDT designed for the purpose of assignment. It is mostly derived from known RDT's like Go-back-n or selective repeat. Firstly putting every packet in a list then adding a checksum, time and packet number for each of the packets and send it to server. After sending them server waits for an answer from server, if server send to client an ACK(acknowledge) it comes with a packet number, it has it's own checksum too, for making sure it is not corrupted as well. If message is ACK server takes the packet number that is ACKed and put it in a uniqely elemented ack_list. If list has already have this packet number server do nothing, since it is probably an unnecessary ACK. If message is NACK, server simply do nothing for this time. And continue to send other packets. After finishing a tour in packet list, server starts a new one and this time sends only the packets that didn't get an ACK. Server keeps doing this process until there is no packet has not received an ACK or the cleint stop responding for 15 seconds. In this case most probable scenario is server sends this ACK's but they got corrupted and client couldn't receive them before the server got all the packets it need and closed.

In server side, server checks the packets check their checksum and send ACK's or NACK's respectively. If a packet arrived correctly server take their packet number and insert in a packets list with it's packet number. When the last packet arrived with <END> token that known the how many packets there is and if server got the all packets in my list, it closes the connection, sort the packets and write it into a file.
