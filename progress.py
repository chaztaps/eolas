import glob

tags = ['deer','sheep','negatives']

for tag in tags:
	images = glob.glob('augmented/{}/*png'.format(tag))
	print(tag,len(images))
#bitch
#bitttttttchhhhh
