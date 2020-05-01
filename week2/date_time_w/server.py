import socket
from datetime import datetime
today = datetime.now()
dict={}
dict["year"], dict["month"], dict["day"], dict["time"] = str(today)[:4], str(today)[5:7], str(today)[8:10], str(today)[
                                                                                                            11:16]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mysocket:

    mysocket.bind(('', 8000))
    mysocket.listen(1)
    print("The server ready to accept commands!")
    connn, addr = mysocket.accept()
    with connn:
        while True:
            data = connn.recv(1024)
            if (data.decode()) in dict.keys():
                connn.sendall(dict[str(data.decode())].encode())
            else:
                if data.decode() == "exit":
                    break
                else:
                    connn.sendall(b"Unrecognized command   " + b"'" + data + b"'")








