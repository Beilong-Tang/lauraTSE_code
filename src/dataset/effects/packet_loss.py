import numpy as np 
def packet_loss_config(
    speech_length, fs, packet_duration_ms, packet_loss_rate, max_continuous_packet_loss
):
    """Returns a list of indices (of packets) that are zeroed out."""

    # speech duration in ms and the number of packets
    speech_duration_ms = speech_length / fs * 1000
    num_packets = int(speech_duration_ms // packet_duration_ms)

    # randomly select the packet loss rate and calculate the packet loss duration
    packet_loss_rate = np.random.uniform(*packet_loss_rate)
    packet_loss_duration_ms = packet_loss_rate * speech_duration_ms

    # calculate the number of packets to be zeroed out
    num_packet_loss = int(round(packet_loss_duration_ms / packet_duration_ms, 0))

    # list of length of each packet loss
    packet_loss_lengths = []
    for _ in range(num_packet_loss):
        num_continuous_packet_loss = np.random.randint(1, max_continuous_packet_loss)
        packet_loss_lengths.append(num_continuous_packet_loss)

        if num_packet_loss - sum(packet_loss_lengths) <= max_continuous_packet_loss:
            packet_loss_lengths.append(num_packet_loss - sum(packet_loss_lengths))
            break

    packet_loss_start_indices = np.random.choice(
        range(num_packets), len(packet_loss_lengths), replace=False
    )
    packet_loss_indices = []
    for idx, length in zip(packet_loss_start_indices, packet_loss_lengths):
        packet_loss_indices += list(range(idx, idx + length))

    return list(set(packet_loss_indices))

def packet_loss(
    speech_sample, fs: int, packet_loss_indices: list, packet_duration_ms: int = 20
):
    for idx in packet_loss_indices:
        start = idx * packet_duration_ms * fs // 1000
        end = (idx + 1) * packet_duration_ms * fs // 1000
        speech_sample[:, start:end] = 0
    return speech_sample

