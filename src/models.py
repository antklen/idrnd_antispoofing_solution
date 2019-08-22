from keras import optimizers
from keras.layers import (Activation, AvgPool1D, AvgPool2D, BatchNormalization,
                          Conv1D, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling1D, GlobalAvgPool2D,
                          GlobalMaxPool2D, GlobalMaxPooling1D, Input,
                          MaxPool2D, MaxPooling1D, concatenate)
from keras.models import Model


def CNN1D(input_shape, output_shape=1, output_activation='sigmoid',
          num_conv=2, num_filters=[64, 64, 64], kernel_size=[3, 3, 3],
          activation='relu', strides=[1, 1, 1], batch_norm=True,
          pooling='max', pool_size=2, conv_dropout=0,
          global_pooling='max', dense_units=[64], dropout=0,
          batch_norm_dense=False,
          optimizer=optimizers.Adam, optimizer_params={'lr': 1e-3},
          loss='binary_crossentropy', metrics=None):

    assert len(num_filters) == len(kernel_size) == len(strides)

    inputs = Input(shape=input_shape, dtype='float32')
    x = inputs

    for i in range(len(num_filters)-1):
        x = conv1d_block(x, num_conv, num_filters[i], kernel_size[i],
                         activation, strides[i], batch_norm,
                         pooling, pool_size, conv_dropout)

    x = conv1d_block(x, num_conv, num_filters[-1], kernel_size[-1],
                     activation, strides[-1], batch_norm,
                     pooling=None, dropout=conv_dropout)

    if global_pooling == 'max':
        x = GlobalMaxPooling1D()(x)
    elif global_pooling == 'avg':
        x = GlobalAveragePooling1D()(x)
    elif global_pooling is None:
        x = Flatten()(x)

    for i in range(len(dense_units)):
        x = Dense(dense_units[i])(x)
        if batch_norm_dense:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    output = Dense(output_shape, activation=output_activation)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=optimizer(**optimizer_params),
                  metrics=metrics)

    return model


def conv1d_block(inputs, num_conv=1, num_filters=64, kernel_size=3,
                 activation='relu', strides=1, batch_norm=True,
                 pooling='max', pool_size=2, dropout=0):

    x = inputs

    for i in range(num_conv):
        x = Conv1D(filters=num_filters, kernel_size=kernel_size,
                   strides=strides)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)

    if pooling == 'max':
        x = MaxPooling1D(pool_size=pool_size)(x)
    elif pooling == 'avg':
        x = AvgPool1D(pool_size=pool_size)(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    return x


def CNN2D(input_shape, output_shape=1, output_activation='sigmoid',
          num_conv=2, num_filters=[64, 64, 64],
          kernel_size=[(3, 3), (3, 3), (3, 3)],
          activation='relu', strides=[1, 1, 1], batch_norm=True,
          pooling='max', pool_size=(2, 2), conv_dropout=0,
          global_pooling='max', dense_units=[64], dropout=0,
          batch_norm_dense=True,
          optimizer=optimizers.Adam, optimizer_params={'lr': 1e-3},
          loss='binary_crossentropy', metrics=None):

    assert len(num_filters) == len(kernel_size) == len(strides)

    inputs = Input(shape=input_shape, dtype='float32')
    x = inputs

    for i in range(len(num_filters)-1):
        x = conv2d_block(x, num_conv, num_filters[i], kernel_size[i],
                         activation, strides[i], batch_norm,
                         pooling, pool_size, conv_dropout)

    x = conv2d_block(x, num_conv, num_filters[-1], kernel_size[-1],
                     activation, strides[-1], batch_norm,
                     pooling=None, dropout=conv_dropout)

    if global_pooling == 'max':
        x = GlobalMaxPool2D()(x)
    elif global_pooling == 'avg':
        x = GlobalAvgPool2D()(x)
    elif global_pooling == 'time':
        time_steps = x.shape[1].value
        x = MaxPool2D(pool_size=(time_steps, 1))(x)
        x = Flatten()(x)
    elif global_pooling == 'freq':
        freq_steps = x.shape[2].value
        x = MaxPool2D(pool_size=(1, freq_steps))(x)
        x = Flatten()(x)
    elif global_pooling == 'time+freq':
        time_steps = x.shape[1].value
        x1 = MaxPool2D(pool_size=(time_steps, 1))(x)
        freq_steps = x.shape[2].value
        x2 = MaxPool2D(pool_size=(1, freq_steps))(x)
        x = concatenate([Flatten()(x1), Flatten()(x2)])
    elif global_pooling is None:
        x = Flatten()(x)

    for i in range(len(dense_units)):
        x = Dense(dense_units[i])(x)
        if batch_norm_dense:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    output = Dense(output_shape, activation=output_activation)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=optimizer(**optimizer_params),
                  metrics=metrics)

    return model


def conv2d_block(inputs, num_conv=1, num_filters=64, kernel_size=(3, 3),
                 activation='relu', strides=(1, 1), batch_norm=True,
                 pooling='max', pool_size=(2, 2), dropout=0):

    x = inputs

    for i in range(num_conv):
        x = Conv2D(filters=num_filters, kernel_size=kernel_size,
                   strides=strides)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)

    if pooling == 'max':
        x = MaxPool2D(pool_size=pool_size)(x)
    elif pooling == 'avg':
        x = AvgPool2D(pool_size=pool_size)(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    return x
