from flask import session, request
from flask_socketio import emit, join_room, leave_room
from .. import socketio

clients = []


# @socketio.on('connect')
# def handle_connect():
#     clients.append(request.sid)


def send_message(client_id, data):
    socketio.emit('output', data, room=client_id)

    print('sending message "{}" to client "{}".'.format(data, client_id))


@socketio.on('joined', namespace='/')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    clients.append(request.sid)
    # print(f'join - {session.get("name")}')
    room = session.get('room')
    join_room(room)
    emit('status', {'msg': session.get('name') + ' has joined the chat',
                    'clients_count': int(len(clients))}, room=room)


@socketio.on('text', namespace='/')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    room = session.get('room')
    emit('message', {'msg': session.get('name') + ':' + message['msg']},
         room=room)


@socketio.on('left', namespace='/')
def left(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room')
    leave_room(room)
    emit('status', {'msg': session.get('name') + ' has left the chat'},
         room=room)


@socketio.on('disconnect')
def handle_disconnect():
    # print(f'leave - {session.get("name")}')
    clients.remove(request.sid)
    room = session.get('room')
    leave_room(room)

    emit('user_leave', {'msg': session.get('name') + ' has left the chat',
                        'clients_count': int(len(clients))}, room=room)
