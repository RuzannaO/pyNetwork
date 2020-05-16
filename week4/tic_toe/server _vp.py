import socket
import configparser
from configparser import ConfigParser
import json
import random

def check_valid_move(a):
    if int(a)>3 or int(a)<0:
        print('incorrect move')
    else:
        return a

def check_if_won(dict):
    if int(dict['1'])+int(dict['4'])+int(dict['7'])==3 or int(dict['2'])+int(dict['5'])+int(dict['8'])==3 or int(dict['3'])+int(dict['6'])+int(dict['9'])==3 or int(dict['1'])+int(dict['5'])+int(dict['9'])==3 or int(dict['3'])+int(dict['5'])+int(dict['7'])==3:
        return('you win')
    else:
         for i in dict.values():
            if i=='':
                pass
            else:
                print('game over')

    #         not complete (checking if alldict values are set)


config = ConfigParser()
config.read('config.ini')
port=int(config["inet"]["port"])
IP=config["inet"]["ip"]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((IP, port))
    sock.listen(1)

    conn, addr = sock.accept()
    # the dictionary dict1 will help us to assign moves to the board - the dictionary "game board'
    dict1={(1,1):1,(1,2):2,(1,3):3,(2,1):4,(2,2):5,(2,3):6,(3,1):7,(3,2):8,(3,3):9}

    with conn:
        game_board = {'1': ' ', '2': ' ', '3': ' ',  '4': ' ', '5': ' ', '6': ' ','7': ' ', '8': ' ', '9': ' '}
        jsonboard=json.dumps(game_board)

        def printBoard():
            print(str(game_board['1']) + '|' + str(game_board['2']) + '|' + str(game_board['3']))
            print('-+-+-')
            print(str(game_board['4']) + '|' + str(game_board['5']) + '|' + str(game_board['6']))
            print('-+-+-')
            print(str(game_board['7']) + '|' + str(game_board['8']) + '|' + str(game_board['9']))
        while True:
            data = conn.recv(1024)

            a = int(data.decode())
            print(check_valid_move(a))
            conn.sendall(str(check_valid_move(a)).encode())
            data = conn.recv(1024)
            b = int(data.decode())
            print(check_valid_move(b))
            conn.sendall(str(check_valid_move(b)).encode())
            game_board[str(dict1[(a, b)])] = 1
            print(game_board)
            print(printBoard())
            jsonboard = json.dumps(game_board)
            conn.sendall(jsonboard.encode())
            # check if won - not complete, data type issue
            check_if_won(game_board)
            # server's move (c,d), not complete
            c=random.randint(1,3)
            d=random.randint(1,3)

            game_board[str(dict1[(c,b)])] = "0"

            print(printBoard())

            conn.sendall(jsonboard.encode())

