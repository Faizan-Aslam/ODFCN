import socketio
import os
import argparse


parser = argparse.ArgumentParser()   
parser.add_argument('--client', required=True, help="client 240,241,242,243,244")
parser.add_argument('--t', required=True, help="total time for taking readings in ms")
parser.add_argument('--timelapse', required=True, help="time between each reading in ms")
args = parser.parse_args()

sio = socketio.Client()
# throwing a message upon connection of the client
def send_sensor_reading():
    while True:
        sio.emit('my_message',{'client':1})
        sio.sleep(3)
# message sent to the client from server during connection
@sio.event
def connect():
    print('connection established')
    sio.start_background_task(send_sensor_reading)
# 
@sio.event
def message_from_server(data):
    action = data
    print(action)
    if action == 4: 
        os.system(f'libcamera-still -t {args.t} --width 1280 --height 960 --timelapse {args.timelapse} client_{args.client}_%03d.jpg')

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect(f'http://141.44.154.{args.client}:5000')
