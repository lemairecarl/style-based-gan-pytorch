#MESSAGE_SIZE = 2048 + 1


def read_stream(stream, size):
    data = b''
    while len(data) < size:
        received_data = stream.read(size)
        data += received_data
    assert len(data) == size
    return data


def write_stream(stream, message, check_len):
    assert len(message) == check_len
    stream.write(message)
