import socket


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8069))


while True:
    user_says = input('your move ')
    client.sendall(user_says.encode())
    data = client.recv(1024)
    print(data.decode())
    # not complete
    if user_says == 'move done':
        pass
    else:
        if user_says == 'exit':
            break

client.shutdown(socket.SHUT_RDWR)
client.close()
