import cv2
import csv
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split

class AutonomousDrive:
    def __init__(self, dataPathRoot, hasHeader):
        self.dataPathRoot = dataPathRoot
        self.hasHeader = hasHeader


    def train(self):
        # Load the data
        centerPaths, leftPaths, rightPaths, steerings = self.loadData()
        imagePaths, steerings = self.combineAllCameraImagePath(centerPaths, leftPaths, rightPaths, steerings, 0.2)
        print('Total Images: {}'.format( len(imagePaths)))

        # Splitting samples and creating generators.
        samples = list(zip(imagePaths, steerings))
        train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        print('Train samples: {}'.format(len(train_samples)))
        print('Validation samples: {}'.format(len(validation_samples)))

        train_generator = self.generator(train_samples, batch_size=32)
        validation_generator = self.generator(validation_samples, batch_size=32)

        # Model creation
        model = self.createNVidiaModel()

        # Compiling and training the model
        model.compile(loss='mse', optimizer='adam')
        history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                             validation_data=validation_generator,nb_val_samples=len(validation_samples),
                                             nb_epoch=3, verbose=1)

        model.save('model.h5')
        print(history_object.history.keys())
        print('Loss')
        print(history_object.history['loss'])
        print('Validation Loss')
        print(history_object.history['val_loss'])


    def loadData(self):
        # Traverse dataPathRoot, load driving training data of all sub directories
        # returns {centerCameraImageFilePaths, leftCameraImageFilePaths, rightCameraImageFilePaths, steerings)

        directories = [x[0] for x in os.walk(self.dataPathRoot)]
        dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
        centerCameraImageFilePaths = []
        leftCameraImageFilePaths = []
        rightCameraImageFilePaths = []
        steerings = []
        for directory in dataDirectories:
            rows = self.loadLogData(directory)
            centerCamPath = []
            leftCamPath = []
            rightCamPath = []
            steers = []
            for line in rows:
                steers.append(float(line[3]))
                centerCamPath.append(directory + '/' + line[0].strip())
                leftCamPath.append(directory + '/' + line[1].strip())
                rightCamPath.append(directory + '/' + line[2].strip())
            centerCameraImageFilePaths.extend(centerCamPath)
            leftCameraImageFilePaths.extend(leftCamPath)
            rightCameraImageFilePaths.extend(rightCamPath)
            steerings.extend(steers)

        return (centerCameraImageFilePaths, leftCameraImageFilePaths, rightCameraImageFilePaths, steerings)

    def combineAllCameraImagePath(self, centerCamImagePath, leftCamImagePath, rightCamImagePath, steering, correction):
        # combine the center, left, right camera images, and apply correction factor for the steering
        # function returns ([imageFilePaths], [steerings])

        imagePaths = []
        allCameraSteerings = []

        imagePaths.extend(centerCamImagePath)
        allCameraSteerings.extend(steering)

        imagePaths.extend(leftCamImagePath)
        allCameraSteerings.extend([x + correction for x in steering])
        #allCameraSteerings.extend(map(lambda x: x + correction, steering))

        imagePaths.extend(rightCamImagePath)
        allCameraSteerings.extend([x - correction for x in steering])
        #allCameraSteerings.extend(map(lambda x: x - correction, steering))

        return (imagePaths, allCameraSteerings)

    def generator(self, sampleData, batch_size=32):
        # Generate images (input X) and steering (output y)   data for training.
        # sampleData are ([imageFilePath], [steerings])
        num_samples = len(sampleData)
        while 1: # Loop forever
            sampleData = sklearn.utils.shuffle(sampleData)
            for offset in range(0, num_samples, batch_size):
                batch_samples = sampleData[offset:offset + batch_size]

                images = []
                steerings = []
                for imagePath, steer in batch_samples:
                    originalImage = cv2.imread(imagePath)
                    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    steerings.append(steer)

                    # Flip the image, and invert the steering in order to normalize the training data (in terms of having evenly distributed left and right turns)
                    images.append(cv2.flip(image,1))
                    steerings.append(steer*-1.0)

                image_X = np.array(images)
                steering_y = np.array(steerings)
                yield sklearn.utils.shuffle(image_X, steering_y)



    def createNVidiaModel(self):
        # Implement NVidia pipline

        #pre-process the image
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

        # trim off the top an bottom of the image so that only the road is factored in training
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))

        # Convolitional layers
        model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model

    # Helper functions

    def loadLogData(self, dataSubDirectory):
        """
        Parse the driving log file with base directory `dataPath`, and return all rows
        If the file include headers, set hasHeader to True
        """
        rows = []
        with open(dataSubDirectory + '/driving_log.csv') as csvFile:
            reader = csv.reader(csvFile)
            if self.hasHeader:
                next(reader, None)
            for row in reader:
                rows.append(row)
        return rows




selfDrive = AutonomousDrive(dataPathRoot='data', hasHeader=True)
selfDrive.train()