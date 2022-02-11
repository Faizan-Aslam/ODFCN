import eventlet
import socketio
import os
import argparse

parser = argparse.ArgumentParser()   
parser.add_argument('--t', required=True, help="total time for taking readings in ms")
parser.add_argument('--timelapse', required=True, help="time between each reading in ms")
args = parser.parse_args()


sio = socketio.Server()
app = socketio.WSGIApp(sio)


count = 0
@sio.event
def connect(sid, environ):
    global count
    count += 1
    print('connect ', sid)
    print('count: '  ,count)
    if count == 5:
        sio.emit('message_from_server', 1)
        os.system(f'libcamera-still -t {args.t} --width 1280 --height 960 --timelapse {args.timelapse} server_244_%03d.jpg')

@sio.event
def my_message(sid, data):
    print('message ', data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
