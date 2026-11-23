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


def test_verdicts_without_a_pole_follow_file_order():
    from emotion_neurons.judge import Verdict, place_verdicts

    expected = [(5, "high", 0), (5, "high", 1), (5, "low", 0), (5, "low", 1), (5, "low", 2)]
    lines = [Verdict(neuron=5, cluster=c, mode="m", scene_windows=0, windows=5, scene="s") for c in (0, 1, 0, 1, 2)]
    placed = place_verdicts(lines, expected)
    assert list(placed) == expected and all(placed[k].pole == k[1] for k in expected)
