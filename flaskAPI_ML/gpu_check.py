import tensorflow as tf


def check_available_gpus(data):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    all_devices = tf.config.list_physical_devices()
    all_devices = [{'Name': d.name, 'Type': d.device_type}
                   for d in all_devices]
    return {'Devices': all_devices}


if __name__ == '__main__':
    print(check_available_gpus(None))
