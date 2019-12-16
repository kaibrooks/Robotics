import robot_client as RC
import time

#rob1 = RC.Robot('Theif', '192.168.1.5', 4243)
#rob2 = RC.Robot('Policeman 1', '192.168.1.4', 4243)
rob3 = RC.Robot('Policeman 2', '192.168.1.7', 4245)

#rob1.connect()
#rob2.connect()
rob3.connect()


for x in range(30):
	#rob2.rotate(5)
	#rob1.rotate(5)
	rob3.rotate(5)
	time.sleep(.01)
time.sleep(.1)
for x in range(30):
	#rob2.rotate(-5)
	#rob1.rotate(-5)
	rob3.rotate(-5)
	time.sleep(.01)
time.sleep(.1)

for x in range(1):
	#rob2.move_forward(3)
	rob3.move_forward(3)
	#rob1.move_forward(3)
	time.sleep(.1)
	#rob2.rotate(-5)
	rob3.rotate(-5)
	#rob1.rotate(-5)
	time.sleep(.1)
