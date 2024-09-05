import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import semlaflow.util.functional as smolF


class TensorFnsTests(unittest.TestCase):
    def test_pairwise_concat_creates_stacked_pairs(self):
        vec_size = 4

        t = torch.rand((3, 2, vec_size))
        pairwise = smolF.pairwise_concat(t)

        expected_shape = (3, 2, 2, 2 * vec_size)
        first_vec = t[0, 0, :]
        second_vec = t[0, 1, :]

        self.assertEqual(expected_shape, pairwise.shape)

        self.assertTrue(torch.equal(first_vec, pairwise[0, 0, 0, :vec_size]))
        self.assertTrue(torch.equal(first_vec, pairwise[0, 0, 1, :vec_size]))

        self.assertTrue(torch.equal(second_vec, pairwise[0, 0, 1, vec_size:]))
        self.assertTrue(torch.equal(second_vec, pairwise[0, 1, 1, vec_size:]))

        self.assertTrue(torch.equal(first_vec, pairwise[0, 1, 0, vec_size:]))
        self.assertTrue(torch.equal(second_vec, pairwise[0, 1, 0, :vec_size]))

    def test_segment_sum_adds_feats_for_segments(self):
        batch_size = 2
        seq_len = 5
        num_feats = 4
        num_segments = 3

        t1 = torch.rand((seq_len, num_feats))
        t2 = torch.rand((seq_len, num_feats))
        data = torch.stack((t1, t2))
        segment_ids = torch.tensor([[0, 1, 1, 0, 2], [2, 2, 2, 0, 0]])

        expected_shape = (batch_size, num_segments, num_feats)

        exp_b0_s0 = t1[0] + t1[3]
        exp_b0_s1 = t1[1] + t1[2]
        exp_b0_s2 = t1[4]

        exp_b1_s0 = t2[3] + t2[4]
        exp_b1_s1 = torch.zeros(num_feats)
        exp_b1_s2 = t2[0] + t2[1] + t2[2]

        segment_sums = smolF.segment_sum(data, segment_ids, num_segments)

        self.assertEqual(expected_shape, segment_sums.shape)

        self.assertTrue(torch.equal(exp_b0_s0, segment_sums[0, 0]))
        self.assertTrue(torch.equal(exp_b0_s1, segment_sums[0, 1]))
        self.assertTrue(torch.equal(exp_b0_s2, segment_sums[0, 2]))

        self.assertTrue(torch.equal(exp_b1_s0, segment_sums[1, 0]))
        self.assertTrue(torch.equal(exp_b1_s1, segment_sums[1, 1]))
        self.assertTrue(torch.equal(exp_b1_s2, segment_sums[1, 2]))


class EdgeFnsTests(unittest.TestCase):
    def test_adj_from_node_mask_correct_adj(self):
        num_nodes = 4

        t1_nodes = torch.tensor([1, 1, 1, 1])
        t2_nodes = torch.tensor([1, 1, 1, 0])
        t3_nodes = torch.tensor([0, 0, 0, 0])
        node_mask = torch.stack((t1_nodes, t2_nodes, t3_nodes))

        exp_shape = (3, num_nodes, num_nodes)
        exp_type = torch.long

        b0_exp = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        b1_exp = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
        b2_exp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        adjacency = smolF.adj_from_node_mask(node_mask)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(b0_exp, adjacency[0].tolist())
        self.assertEqual(b1_exp, adjacency[1].tolist())
        self.assertEqual(b2_exp, adjacency[2].tolist())

    def test_edges_from_adj_correct_edges(self):
        t1 = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [2, -1, 0, 0]])
        t2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]])
        adjacency = torch.stack((t1, t2))

        exp_shape = (2, 8)
        exp_type = torch.long

        exp_edges_i_b0 = [0, 0, 0, 0, 1, 1, 3, 3]
        exp_edges_j_b0 = [0, 1, 2, 3, 0, 2, 0, 1]

        exp_edges_i_b1 = [1, 3, 0, 0, 0, 0, 0, 0]
        exp_edges_j_b1 = [3, 3, 0, 0, 0, 0, 0, 0]

        exp_mask_b0 = [1, 1, 1, 1, 1, 1, 1, 1]
        exp_mask_b1 = [1, 1, 0, 0, 0, 0, 0, 0]

        (edge_is, edge_js), edge_mask = smolF.edges_from_adj(adjacency)

        # Check shapes
        self.assertEqual(exp_shape, edge_is.shape)
        self.assertEqual(exp_shape, edge_js.shape)
        self.assertEqual(exp_shape, edge_mask.shape)

        # Check types
        self.assertEqual(exp_type, edge_is.dtype)
        self.assertEqual(exp_type, edge_js.dtype)
        self.assertEqual(exp_type, edge_mask.dtype)

        # Check edge indices
        self.assertEqual(exp_edges_i_b0, edge_is[0].tolist())
        self.assertEqual(exp_edges_j_b0, edge_js[0].tolist())
        self.assertEqual(exp_edges_i_b1, edge_is[1].tolist())
        self.assertEqual(exp_edges_j_b1, edge_js[1].tolist())

        # Check mask
        self.assertEqual(exp_mask_b0, edge_mask[0].tolist())
        self.assertEqual(exp_mask_b1, edge_mask[1].tolist())

    def test_adj_from_edges_correct_adj(self):
        num_nodes = 4

        edges = torch.tensor([[0, 0, 1], [0, 2, 2], [1, 0, 1], [1, 3, 0], [2, 2, 3], [3, 1, 1]])

        exp_shape = (num_nodes, num_nodes)
        exp_type = torch.long

        exp_adj = [[1, 0, 2, 0], [1, 0, 0, 0], [0, 0, 3, 0], [0, 1, 0, 0]]

        edge_indices = edges[:, :2]
        edge_types = edges[:, 2]

        adjacency = smolF.adj_from_edges(edge_indices, edge_types, num_nodes)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(exp_adj, adjacency.tolist())

    def test_edges_from_nodes_fully_connected(self):
        num_nodes = 4

        coords_b0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 3.0, 1.0], [4.0, -2.0, -3.0]])
        coords_b1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [-1.0, -5.0, 2.0]])
        coords = torch.stack((coords_b0, coords_b1))

        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        exp_shape = (2, num_nodes, num_nodes)
        exp_type = torch.long

        exp_adj_b0 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        exp_adj_b1 = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        adjacency = smolF.edges_from_nodes(coords, node_mask=mask)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(exp_adj_b0, adjacency[0].tolist())
        self.assertEqual(exp_adj_b1, adjacency[1].tolist())

    def test_edges_from_nodes_correct_neighbours(self):
        num_nodes = 4

        coords_b0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 3.0, 1.0], [4.0, -2.0, -3.0]])
        coords_b1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [-1.0, -5.0, 2.0]])
        coords = torch.stack((coords_b0, coords_b1))

        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        exp_shape = (2, num_nodes, num_nodes)
        exp_type = torch.long

        exp_adj_b0 = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
        exp_adj_b1 = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        adjacency = smolF.edges_from_nodes(coords, k=2, node_mask=mask)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(exp_adj_b0, adjacency[0].tolist())
        self.assertEqual(exp_adj_b1, adjacency[1].tolist())


class SparseFnsTests(unittest.TestCase):
    def test_gather_edge_features(self):
        feats_b0 = torch.tensor(
            [
                [[0.5, 1.0], [0.1, -0.5], [5.0, -2.0], [-0.1, 0.8]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                [[0.6, -0.2], [0.5, -2.0], [-7.0, 4.0], [5.0, 6.0]],
            ]
        )
        feats_b1 = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                [[-1.0, -2.0], [-2.0, -3.0], [-3.0, -4.0], [-4.0, -5.0]],
                [[0.6, 0.9], [0.3, 0.2], [0.1, -0.7], [-0.5, 0.9]],
                [[1.5, -2.8], [6.3, 2.9], [5.8, 9.1], [0.4, -3.7]],
            ]
        )
        feats = torch.stack((feats_b0, feats_b1))

        adj_1 = torch.tensor(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            ]
        ).long()

        adj_2 = torch.tensor(
            [
                [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0]],
                [[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
            ]
        ).long()

        exp_feats_1_b0 = [[[0.5, 1.0]], [[3.0, 4.0]], [[13.0, 14.0]], [[5.0, 6.0]]]
        exp_feats_1_b1 = [[[0.7, 0.8]], [[-4.0, -5.0]], [[-0.5, 0.9]], [[0.4, -3.7]]]
        exp_feats_1 = [exp_feats_1_b0, exp_feats_1_b1]

        exp_feats_2_b0 = [
            [[0.1, -0.5], [5.0, -2.0]],
            [[1.0, 2.0], [7.0, 8.0]],
            [[11.0, 12.0], [15.0, 16.0]],
            [[0.6, -0.2], [0.5, -2.0]],
        ]
        exp_feats_2_b1 = [
            [[0.3, 0.4], [0.7, 0.8]],
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[0.6, 0.9], [0.3, 0.2]],
            [[6.3, 2.9], [5.8, 9.1]],
        ]
        exp_feats_2 = [exp_feats_2_b0, exp_feats_2_b1]

        gathered_1 = smolF.gather_edge_features(feats, adj_1)
        gathered_2 = smolF.gather_edge_features(feats, adj_2)

        np.testing.assert_almost_equal(exp_feats_1, gathered_1.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_feats_2, gathered_2.tolist(), decimal=5)


class GeometryFnsTests(unittest.TestCase):
    def test_calc_distance_without_edges(self):
        num_nodes = 4

        coords_b0 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [5.0, -1.0, -2.0]])
        coords_b1 = torch.tensor([[0.5, 1.0, -0.25], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        coords = torch.stack((coords_b0, coords_b1))

        exp_shape = (2, num_nodes, num_nodes)
        exp_type = torch.float

        exp_b0 = [[0.0, 0.0, 5.0, 29.0], [0.0, 0.0, 5.0, 29.0], [5.0, 5.0, 0.0, 46.0], [29.0, 29.0, 46.0, 0.0]]
        exp_b1 = [
            [0.0, 1.8125, 1.3125, 1.3125],
            [1.8125, 0.0, 3.0, 3.0],
            [1.3125, 3.0, 0.0, 0.0],
            [1.3125, 3.0, 0.0, 0.0],
        ]

        sqrd_dists = smolF.calc_distances(coords, sqrd=True)
        dists = torch.sqrt(sqrd_dists)

        self.assertEqual(exp_shape, sqrd_dists.shape)
        self.assertEqual(exp_type, sqrd_dists.dtype)

        np.testing.assert_almost_equal(exp_b0, sqrd_dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_b1, sqrd_dists[1].tolist(), decimal=5)

        np.testing.assert_almost_equal(np.sqrt(exp_b0).tolist(), dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(np.sqrt(exp_b1).tolist(), dists[1].tolist(), decimal=5)

    def test_calc_distances_from_edges(self):
        num_edges = 8

        coords_b0 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [5.0, -1.0, -2.0]])
        coords_b1 = torch.tensor([[0.5, 1.0, -0.25], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        coords = torch.stack((coords_b0, coords_b1))

        edge_is = torch.tensor([[0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 1, 1, 0, 0, 0]])
        edge_js = torch.tensor([[0, 1, 2, 3, 0, 2, 0, 0], [0, 1, 2, 2, 2, 0, 0, 0]])
        edges = (edge_is, edge_js)

        exp_shape = (2, num_edges)
        exp_type = torch.float

        exp_b0 = [0.0, 0.0, 5.0, 29.0, 0.0, 0.0, 5.0, 5.0]
        exp_b1 = [0.0, 1.8125, 1.3125, 3.0, 3.0, 0.0, 0.0, 0.0]

        sqrd_dists = smolF.calc_distances(coords, edges=edges, sqrd=True)
        dists = torch.sqrt(sqrd_dists)

        self.assertEqual(exp_shape, sqrd_dists.shape)
        self.assertEqual(exp_type, sqrd_dists.dtype)

        np.testing.assert_almost_equal(exp_b0, sqrd_dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_b1, sqrd_dists[1].tolist(), decimal=5)

        np.testing.assert_almost_equal(np.sqrt(exp_b0).tolist(), dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(np.sqrt(exp_b1).tolist(), dists[1].tolist(), decimal=5)

    def test_calc_com_correct_centre(self):
        coords_b0 = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.0], [-4.0, 2.0, 2.0], [3.0, -5.0, -5.0]])
        coords_b1 = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.0], [-4.0, 2.0, 1.0], [3.0, -5.0, -5.0]])
        coords_b2 = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.0], [-4.0, 2.0, 1.0], [3.0, -5.0, -5.0]])
        coords = torch.stack((coords_b0, coords_b1, coords_b2))
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]])

        exp_shape = (3, 1, 3)
        exp_type = torch.float

        exp_com_b0 = [0.5, -1.0, -0.5]
        exp_com_b1 = [1.5, -0.5, 0.5]
        exp_com_b2 = [np.nan, np.nan, np.nan]

        com = smolF.calc_com(coords, node_mask=mask)

        self.assertEqual(exp_shape, com.shape)
        self.assertEqual(exp_type, com.dtype)

        self.assertEqual(exp_com_b0, com[0, 0, :].tolist())
        self.assertEqual(exp_com_b1, com[1, 0, :].tolist())
        np.testing.assert_equal(exp_com_b2, com[2, 0, :].tolist())

    def test_rotate_rotates_all_coords_correctly(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 2.0, 0.5]])

        rot1 = [np.pi / 2, 0.0, 0.0]
        rot2 = [0.0, np.pi / 2, 0.0]
        rot3 = [0.0, 0.0, np.pi / 2]
        rot4 = [np.pi / 2, np.pi, np.pi / 2]
        rot5 = [-np.pi / 2, 0.0, 2 * np.pi]

        exp_coords_1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, -0.5, 2.0]]
        exp_coords_2 = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 2.0, 1.0]]
        exp_coords_3 = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-2.0, -1.0, 0.5]]
        exp_coords_4 = [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.5, 1.0, -2.0]]
        exp_coords_5 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.5, -2.0]]

        rotated_1 = smolF.rotate(coords, rot1)
        rotated_2 = smolF.rotate(coords, rot2)
        rotated_3 = smolF.rotate(coords, rot3)
        rotated_4 = smolF.rotate(coords, rot4)
        rotated_5 = smolF.rotate(coords, rot5)

        np.testing.assert_almost_equal(exp_coords_1, rotated_1.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_2, rotated_2.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_3, rotated_3.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_4, rotated_4.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_5, rotated_5.tolist(), decimal=5)

    def test_rotate_agrees_with_scipy(self):
        coords = torch.rand((10, 3))

        angles_1 = (np.random.rand(3) * np.pi * 2).tolist()
        angles_2 = (np.random.rand(3) * np.pi * 2).tolist()
        angles_3 = (np.random.rand(3) * np.pi * 2).tolist()

        rot1 = Rotation.from_euler("xyz", angles_1)
        rot2 = Rotation.from_euler("xyz", angles_2)
        rot3 = Rotation.from_euler("xyz", angles_3)

        exp_coords_1 = rot1.apply(coords.tolist())
        exp_coords_2 = rot2.apply(coords.tolist())
        exp_coords_3 = rot3.apply(coords.tolist())

        rotated_1 = smolF.rotate(coords, angles_1)
        rotated_2 = smolF.rotate(coords, angles_2)
        rotated_3 = smolF.rotate(coords, angles_3)

        np.testing.assert_almost_equal(exp_coords_1, rotated_1, decimal=5)
        np.testing.assert_almost_equal(exp_coords_2, rotated_2, decimal=5)
        np.testing.assert_almost_equal(exp_coords_3, rotated_3, decimal=5)


if __name__ == "__main__":
    unittest.main()
