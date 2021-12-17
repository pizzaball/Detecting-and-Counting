class TrackableObject:
	def __init__(self, objectID, centre):
		# store the object ID, then initialize a list of centres
		# using the current centre
		self.objectID = objectID
		self.centre = [centre]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False