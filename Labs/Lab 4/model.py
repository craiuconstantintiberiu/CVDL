import tensorflow
from tensorflow.keras.layers import Conv2D, Dense, Input, GlobalAvgPool2D
from tensorflow.keras.models import Model

from data_loader import DataGenerator


def resnet_block(input, filter_size=3, no_filters=16):
    layer1 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same")(input)
    layer2 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same")(layer1)
    return tensorflow.keras.layers.Add()([input, layer2])


def build_mini_resnet(input_size, num_classes):
    inputs = Input(shape=input_size)
    x = Conv2D(kernel_size=3, filters=16, strides=2)(inputs)
    x = resnet_block(x)
    x = resnet_block(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(num_classes)(x)
    return Model(inputs=inputs, outputs=x, name="mini_resnet")


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    train_generator = DataGenerator("./images", 32, input_shape, 37)


    label_names = train_generator.class_names

    model = build_mini_resnet(input_shape, 37)
    model.summary()
    model.compile(optimizer='adam', loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x=train_generator, batch_size=32, epochs=10)
    model.save("modelLab4.h5")
