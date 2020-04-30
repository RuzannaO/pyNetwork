import socket
from datetime import datetime
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mysocket:

    mysocket.bind(('', 8000))
    mysocket.listen(1)
    print("Server ready to accept commands!")
    connn, addr = mysocket.accept()
    with connn:
        while True:
            data = connn.recv(1024)
            if data.decode() == "exit":
                break
            if data.decode() == "year":
                connn.sendall(str(datetime.now())[0:4].encode())
            else:
                if data.decode() == "month":
                    connn.sendall(str(datetime.now())[5:7].encode())
                else:
                    if data.decode() == "day":
                        connn.sendall(str(datetime.now())[8:10].encode())
                    else:
                        if data.decode() == "time":
                            connn.sendall(str(datetime.now())[11:16].encode())
                        else:
                            connn.sendall(b"Unrecognized command   " + b"'" + data + b"'")









