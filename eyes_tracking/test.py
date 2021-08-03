from pygame import mixer
mixer.init()
explosionSound = mixer.Sound("song.mp3")

while True:
	
	print("Press 'p' to pause, 'r' to resume")
	print("Press 'e' to exit the program")
	query = input(" ")
	print('running')
	
	explosionSound.play()
	# if query == 'p':

	if query == 'e':

		# Stop the mixer
		explosionSound.stop()
		# break
