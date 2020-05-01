import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientsocket:
    clientsocket.connect(("127.0.0.1", 8000))
    while True:
        user_says = input()
        if user_says == "exit":
            clientsocket.sendall(user_says.encode())
            break
        else:
            clientsocket.sendall(user_says.encode())
            data = clientsocket.recv(1024)
            print(str(data.decode()))
