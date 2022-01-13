import glob
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape, num_classes,
                 shuffle=True):
        # TODO your initialization
        # you might want to store the parameters into class variables
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        # load the data from the root directory
        self.class_names = []
        self.data, self.labels = self.get_data(db_dir)
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def get_data(self, root_dir):
        """"
        Loads the paths to the images and their corresponding labels from the database directory
        """
        # TODO your code here
        paths = glob.glob(root_dir + "/*.jpg")
        labels = ["_".join(os.path.basename(path).split("_")[:-1]) for path in paths]
        self.class_names = list(set(labels))
        sorted(self.class_names)

        self.data = paths
        self.labels = np.array([self.class_names.index(label) for label in labels])
        return self.data, self.labels

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        batch_x = [] # TODO load the image from batch_indices
        for i in batch_indices:
            try:
                image = cv2.imread(self.data[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.square_image(image)
                image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                batch_x.append(image)
            except :
                batch_x.append(np.zeros(self.input_shape))
        batch_y = self.labels[batch_indices] # TODO load the corresponding labels of the images you loaded
        # optionally you can use: batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return np.asarray(batch_x), np.asarray(batch_y)

    def square_image(self, image):
        width_pad = 0
        height_pad = 0
        if image.shape[0] > image.shape[1]:
            width_pad = (image.shape[0] - image.shape[1])//2
        else:
            height_pad = (image.shape[1] - image.shape[0])//2
        return np.pad(image, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)), mode="edge")

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            # TODO shuffle data
            # you might find np.random.shuffle useful here
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    train_generator = DataGenerator("./images", 32, (400, 400, 3), 37)
    label_names = train_generator.class_names
    assert len(label_names) == 37
    batch_x, batch_y = train_generator[0]

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=[16, 9])
    for i in range(len(axes)):
        axes[i].set_title(label_names[batch_y[i]])
        axes[i].imshow(batch_x[i])
    plt.show()
