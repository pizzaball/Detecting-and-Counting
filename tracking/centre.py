from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentreTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID with two ordered
        # dictionaries used to track of mapping a given object
        # ID to its centre and number of consecutive frames it hasd
        # been marked as "disappeared"
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centres to associate
        # an object
        # if the distance is larger than this maximum
        # distance we will start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centre):
        # when registering an object we use the next available object
        # ID to store the centre
        self.objects[self.nextObjectID] = centre
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centres or tracking info
            # to update
            return self.objects

        # initialize an array of input centres for the current frame
        inputCentres = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to obtain the centre
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentres[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentres)):
                self.register(inputCentres[i])

        # otherwise, are are currently tracking objects and we need to
        # try to match the input centres to existing object
        # centres
        else:
            # grab the set of object IDs and corresponding centres
            objectIDs = list(self.objects.keys())
            objectCentres = list(self.objects.values())

            # compute the distance between each pair of object
            # centres and input centres
            # our goal will be to match an input centre to an existing
            # object centre
            D = dist.cdist(np.array(objectCentres), inputCentres)

            # in order to perform this matching we must 1) find the
            # smallest value in each row, 2) sort the row
            # indexes based on their minimum values that the row
            # with the smallest value as at the 'front' of the index list
            rows = D.min(axis=1).argsort()

            # next we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already surveyed
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centres is greater than
                # the maximum distance, do not associate the two
                # centres to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centre, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentres[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have not yet surveyed
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centres is
            # equal or greater than the number of input centres
            # we need to check and see if some of these objects
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centres is greater
            # than the number of existing object centres we need to
            # register each new input centre as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentres[col])

        # return the set of trackable objects
        return self.objects