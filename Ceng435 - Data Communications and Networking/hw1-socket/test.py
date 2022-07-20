chunks = []
while(True):
	with open("transfer_file_TCP.txt", "rb") as f:
		bytes_read = f.read(1000)
		if not bytes_read:
			break
		chunks.append(bytes_read)
		print(chunks)
print("data written to array")