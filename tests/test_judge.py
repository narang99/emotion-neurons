from emotion_neurons.judge import cluster_keys, instructions, parse_verdicts, read_verdict


def test_verdicts_take_the_emotion_named_count():
    reply = 'x {"neuron": 7, "pole": "low", "cluster": 2, "mode": "m", "angry_windows": 4, "windows": 5, "scene": "s"} y'
    (v,) = parse_verdicts(reply)
    assert (v.neuron, v.pole, v.cluster, v.scene_windows) == (7, "low", 2, 4)


def test_old_happy_files_still_read():
    v = read_verdict('{"neuron": 1, "pole": "high", "cluster": 0, "mode": "m", "happy_windows": 3, "windows": 5, "scene": "s"}')
    assert v.scene_windows == 3


def test_instructions_name_the_emotion_key():
    assert '"angry_windows"' in instructions("angry") and "happy" not in instructions("angry").lower().replace("unhappy", "")


def test_cluster_keys_skip_unclustered_groups():
    text = "##### neuron 5\n=== high pole (p97)\n--- cluster 0\n  [1.0] a\n--- unclustered (sample)\n=== low pole (p3)\n--- cluster 1\n"
    assert cluster_keys(text) == [(5, "high", 0), (5, "low", 1)]
