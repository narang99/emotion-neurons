import torch

from emotion_neurons.vectors import category_means, diff_of_means

torch.manual_seed(0)


def test_categories_count_equally_whatever_their_sizes():
    x = torch.randn(6, 3, dtype=torch.float64)
    label = torch.tensor([0, 1, 1, 1, 1, 1])
    means = category_means(x, label, 2, torch.ones(6, dtype=torch.bool))
    torch.testing.assert_close(diff_of_means(means, "all")[0], (x[0] - x[1:].mean(0)) / 2)


def test_among_and_out_of_range_labels_are_left_out():
    x = torch.randn(5, 3, dtype=torch.float64)
    label = torch.tensor([0, 0, 1, -1, 1])
    among = torch.tensor([True, False, True, True, True])
    means = category_means(x, label, 2, among)
    torch.testing.assert_close(means, torch.stack([x[0], x[[2, 4]].mean(0)]))


def test_rest_of_two_categories_is_one_mean_minus_the_other():
    means = torch.randn(2, 4, 3, dtype=torch.float64)
    assert torch.equal(diff_of_means(means, "rest")[0], means[0] - means[1])


def test_all_is_rest_shortened_by_the_category_count():
    means = torch.randn(5, 4, dtype=torch.float64)
    torch.testing.assert_close(diff_of_means(means, "all"), diff_of_means(means, "rest") * 4 / 5)
