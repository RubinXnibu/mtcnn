import tensorflow as tf


"""
tf-record items:
@image_feature
@landmark: [x, y]
@image_name
"""


def _parse_data(proto):
    feature = tf.parse_single_example(
        proto,
        features={
            "image": tf.FixedLenFeature((), tf.string),
            "label": tf.FixedLenFeature(136, tf.float32)
        }
    )
    print(feature)
    image = tf.image.decode_image(feature['image'])
    image = tf.reshape(image, [128, 128, 3])
    label = feature['label']
    return {'x': image}, label


def train_input_fn(record_path, batch_size):
    dataset = tf.data.TFRecordDataset(record_path)
    dataset = dataset.map(_parse_data)
    # iterator = dataset.make_one_shot_iterator()
    # next_elem = iterator.get_next()
    # a = tf.Session().run(next_elem)
    # print(a[0].shape, a[1].shape)

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    return features, label


def make_tf_record_example():
    image = tf.Session().run(tf.image.encode_jpeg(tf.zeros([128, 128, 3], dtype=tf.uint8)))
    features = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]*136)),
        "image_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=['*.img'.encode('utf-8')]))
    }

    writer = tf.python_io.TFRecordWriter('example.record')
    for _ in range(1000):
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    make_tf_record_example()
    # train_input_fn('./example.record', 60)
