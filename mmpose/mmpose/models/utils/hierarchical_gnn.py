import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(EdgeConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x_bk_c: torch.Tensor, edge_index_batched: torch.Tensor):
        """
        Batched EdgeConv.
        Args:
        - x_bk_c: Node features for the batch, shape [B*K, C_in] (K=num_nodes_per_graph)
        - edge_index_batched: Batched edge_index, shape [2, B*num_edges_per_graph]
                               Indices should correspond to the flattened x_bk_c.
        Returns:
        - x_out: Aggregated node features, shape [B*K, C_out]
        """
        num_total_nodes = x_bk_c.size(0) # B*K
        
        if edge_index_batched.numel() == 0: # Handle no edges case
            return torch.zeros(num_total_nodes, self.conv[0].out_channels, device=x_bk_c.device)

        # Get source and target node features based on batched edge_index
        row, col = edge_index_batched
        x_i = x_bk_c[row]  # [B*num_edges, C_in]
        x_j = x_bk_c[col]  # [B*num_edges, C_in]

        # Compute edge features: e_ij = concat(x_i, x_j - x_i)
        edge_features_cat = torch.cat((x_i, x_j - x_i), dim=1)  # [B*num_edges, 2 * C_in]
        # [B*num_edges, 2 * C_in, 1, 1]
        edge_features_conv_input = edge_features_cat.unsqueeze(-1).unsqueeze(-1)

        # Through convolution layers
        # [B*num_edges, C_out, 1, 1]
        processed_edge_features = self.conv(edge_features_conv_input)
        # [B*num_edges, C_out]
        processed_edge_features = processed_edge_features.squeeze(-1).squeeze(-1)

        # Initialize output tensor for all nodes in the batch
        x_out = torch.zeros(num_total_nodes, processed_edge_features.size(1), 
                            dtype=x_bk_c.dtype, device=x_bk_c.device) # [B*K, C_out]

        # Sum edge features for each source node (row in edge_index)
        # row contains the flattened indices for source nodes across the batch
        x_out.index_add_(0, row, processed_edge_features)

        return x_out

class AttentionModule(nn.Module):
    def __init__(self, in_channels, num_layers, keypoint_groups,
                 num_total_keypoints_k: int): # K from the original graph
        super(AttentionModule, self).__init__()
        self.keypoint_groups = keypoint_groups # List of lists of indices (0 to K-1)
        self.num_groups = len(keypoint_groups)
        self.num_total_keypoints_k = num_total_keypoints_k


        inter_channels = in_channels // 4
        if inter_channels == 0: inter_channels = 1 # Ensure inter_channels is at least 1

        # These will operate on [B*K, C] or [B*num_groups, C]
        self.conv_down_linear = nn.Linear(in_channels, inter_channels)
        self.bn_down = nn.BatchNorm1d(inter_channels)
        
        # EdgeConv for the graph of groups
        self.edge_conv_groups = EdgeConv(inter_channels, inter_channels) # k is unused

        self.aggregate_linear = nn.Linear(inter_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

        # Precompute edge_index for the fully-connected graph of groups
        if self.num_groups > 0:
            self.group_graph_edge_index = self._build_fully_connected_edge_index(
                self.num_groups, torch.device('cpu') # build on CPU, move to device in forward
            )
        else:
            self.group_graph_edge_index = torch.empty((2,0), dtype=torch.long, device=torch.device('cpu'))


    def _build_fully_connected_edge_index(self, num_nodes, device):
        if num_nodes <= 1:
            return torch.empty((2,0), dtype=torch.long, device=device)
        row = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, num_nodes).view(-1)
        col = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(num_nodes, 1).view(-1)
        # Remove self-loops for typical fully connected layer in GCNs unless desired
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        return edge_index

    def forward(self, x_bk_c: torch.Tensor, B: int, K: int):
        """
        Batched AttentionModule.
        Args:
        - x_bk_c: Node features [B*K, C_in]
        - B: Batch size
        - K: Number of keypoints per graph
        Returns:
        - out: Attended node features [B*K, C_in]
        """
        C_in = x_bk_c.size(-1)
        device = x_bk_c.device

        x_b_k_cin = x_bk_c.view(B, K, C_in) # Original features for later modulation

        if self.num_groups == 0 or not self.keypoint_groups:
            # No groups defined, return original features (or multiply by ones)
            return x_bk_c # x_b_k_cin.reshape(B*K, C_in)

        # Downsample features
        x_down_flat = self.conv_down_linear(x_bk_c) # [B*K, C_inter]
        x_down_flat = self.bn_down(x_down_flat)
        x_down_flat = F.silu(x_down_flat, inplace=True)

        x_down_b_k_cinter = x_down_flat.view(B, K, -1) # [B, K, C_inter]

        sampled_group_means_list = []
        valid_groups_exist = False
        for i in range(self.num_groups):
            current_group_indices = [idx for idx in self.keypoint_groups[i] if 0 <= idx < K] # Filter by current K
            if not current_group_indices:
                # Handle empty or out-of-bounds group by appending a zero vector of correct shape
                # to maintain num_groups for EdgeConv graph structure.
                sampled_group_means_list.append(torch.zeros(B, 1, x_down_b_k_cinter.size(-1),
                                                            device=device, dtype=x_down_b_k_cinter.dtype))
                continue

            valid_groups_exist = True
            group_features_b_gs_c = x_down_b_k_cinter[:, current_group_indices, :] # [B, current_group_size, C_inter]
            group_mean_b_1_c = group_features_b_gs_c.mean(dim=1, keepdim=True) # [B, 1, C_inter]
            sampled_group_means_list.append(group_mean_b_1_c)

        if not valid_groups_exist: # All defined groups were empty or invalid for current K
             return x_bk_c

        x_sampled_b_g_cinter = torch.cat(sampled_group_means_list, dim=1) # [B, num_groups, C_inter]

        # Flatten for batched EdgeConv on groups
        x_sampled_flat_bg_cinter = x_sampled_b_g_cinter.reshape(B * self.num_groups, -1) # [B*num_groups, C_inter]

        # Batched edge_index for the graph of groups
        batched_group_edge_index_list = []
        single_group_edge_index = self.group_graph_edge_index.to(device)
        if single_group_edge_index.numel() > 0:
            for i in range(B):
                batched_group_edge_index_list.append(single_group_edge_index + i * self.num_groups)
            batched_group_edge_index = torch.cat(batched_group_edge_index_list, dim=1)
        else: # No edges if num_groups <=1
            batched_group_edge_index = single_group_edge_index

        att_flat_bg_cinter = self.edge_conv_groups(x_sampled_flat_bg_cinter, batched_group_edge_index) # [B*num_groups, C_inter]

        # Aggregate and apply sigmoid (this is where attention scores are formed for groups)
        group_att_scores_flat_bg_cin = self.aggregate_linear(att_flat_bg_cinter) # [B*num_groups, C_in]
        group_att_scores_flat_bg_cin = self.sigmoid(group_att_scores_flat_bg_cin)

        group_att_scores_b_g_cin = group_att_scores_flat_bg_cin.view(B, self.num_groups, C_in)
        final_keypoint_att_scores_b_k_cin = torch.ones(B, K, C_in, device=device, dtype=x_b_k_cin.dtype)

        sum_of_att_for_kps_b_k_cin = torch.zeros(B, K, C_in, device=device, dtype=x_b_k_cin.dtype)
        kp_membership_count_b_k_1 = torch.zeros(B, K, 1, device=device, dtype=x_b_k_cin.dtype)

        for group_idx, group_def in enumerate(self.keypoint_groups):
            # Get the attention score for this specific group: [B, 1, C_in]
            current_group_att_score_b_1_cin = group_att_scores_b_g_cin[:, group_idx:group_idx+1, :]

            for kp_original_idx in group_def:
                sum_of_att_for_kps_b_k_cin[:, kp_original_idx, :] += current_group_att_score_b_1_cin.squeeze(1)
                kp_membership_count_b_k_1[:, kp_original_idx, :] += 1

        grouped_kps_mask = kp_membership_count_b_k_1 > 0  # Shape [B, K, 1], boolean

        safe_counts = torch.where(grouped_kps_mask, 
                                  kp_membership_count_b_k_1, 
                                  torch.ones_like(kp_membership_count_b_k_1))

        calculated_avg_scores = sum_of_att_for_kps_b_k_cin / safe_counts # Shape [B, K, C_in]

        update_mask_b_k_cin = grouped_kps_mask.expand_as(final_keypoint_att_scores_b_k_cin)

        final_keypoint_att_scores_b_k_cin = torch.where(
            update_mask_b_k_cin,
            calculated_avg_scores,
            final_keypoint_att_scores_b_k_cin
        )

        out_b_k_cin = x_b_k_cin * final_keypoint_att_scores_b_k_cin

        return out_b_k_cin.reshape(B*K, C_in)


class HGNNModule(nn.Module):
    def __init__(self, gcn_cfg, keypoint_connections, keypoint_groups=None):
        super(HGNNModule, self).__init__()
        self.num_layers = gcn_cfg['num_layers']
        self.input_dim = gcn_cfg['input_dim'] # This is C for the input keypoint_embeddings [B,K,C]
        self.hidden_dim = gcn_cfg['hidden_dim'] # Internal processing and output dim C'
        self.k_edgeconv = gcn_cfg.get('k', 3)
        
        self.keypoint_connections = keypoint_connections # List of [src, dst] tuples
        self.defined_groups = keypoint_groups

        # Input projection
        self.conv_input = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn_input = nn.BatchNorm1d(self.hidden_dim) # Operates on [N, C]

        # GCN layers
        self.edgeconv_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.edgeconv_layers.append(EdgeConv(self.hidden_dim, self.hidden_dim, k=self.k_edgeconv))
            _k = 0
            if self.keypoint_connections:
                flat_indices = [idx for pair in self.keypoint_connections for idx in pair]
                if flat_indices:
                    _k = max(flat_indices) + 1
                else:
                      pass

            self.attention_modules.append(
                 AttentionModule(self.hidden_dim, self.num_layers, 
                                 keypoint_groups, # Groups are from connections
                                 _k if _k > 0 else 1 # Pass inferred K
                                )
            )
            self.bn_layers.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Precompute the single-sample edge_index
        self._single_sample_edge_index = self._build_single_sample_edge_index(
            # Determine K based on connections or expect it as param
            _k if _k > 0 else 1,
            keypoint_connections, 
            torch.device('cpu')
        )

    def _build_single_sample_edge_index(self, num_nodes_k: int, connections: list, device: torch.device):
        edge_list = []
        if connections:
            for src_idx, dst_idx in connections:
                if 0 <= src_idx < num_nodes_k and 0 <= dst_idx < num_nodes_k:
                    edge_list.append([src_idx, dst_idx])
                    edge_list.append([dst_idx, src_idx]) # Undirected
        
        if not edge_list: # Handle K=1 or no connections
            return torch.empty((2,0), dtype=torch.long, device=device)
            
        return torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()

    def _get_batched_edge_index(self, B: int, K: int, device: torch.device) -> torch.Tensor:
        if self._single_sample_edge_index.numel() == 0:
            return self._single_sample_edge_index.to(device)
        
        edge_indices_list = [self._single_sample_edge_index.to(device) + i * K for i in range(B)]
        return torch.cat(edge_indices_list, dim=1)


    def forward(self, keypoint_embeddings_b_k_c: torch.Tensor):
        """
        Batched forward pass for GHAGCNBlockModule.
        Args:
        - keypoint_embeddings_b_k_c: Input node features, shape [B, K, C_input]
        Returns:
        - x_out_b_k_c: Output node features, shape [B, K, C_hidden]
        """
        B, K, C_input = keypoint_embeddings_b_k_c.shape
        device = keypoint_embeddings_b_k_c.device

        # Reshape for nn.Linear and nn.BatchNorm1d: [B, K, C_input] -> [B*K, C_input]
        x_flat_bk_c = keypoint_embeddings_b_k_c.reshape(B * K, C_input)

        x_flat_bk_h = self.conv_input(x_flat_bk_c)  # [B*K, C_hidden]
        x_flat_bk_h = self.bn_input(x_flat_bk_h)
        x_flat_bk_h = F.silu(x_flat_bk_h, inplace=True)

        # Get batched edge_index
        # K (num_nodes_per_graph) is determined from input shape
        batched_edge_index = self._get_batched_edge_index(B, K, device)

        # Apply GCN layers
        for i in range(self.num_layers):
            x_residual_flat = x_flat_bk_h
            
            x_ec_flat = self.edgeconv_layers[i](x_flat_bk_h, batched_edge_index) # [B*K, C_hidden]
            x_bn_flat = self.bn_layers[i](x_ec_flat) # [B*K, C_hidden]
            
            # AttentionModule expects [B*K, C], B, K
            x_att_flat = self.attention_modules[i](x_bn_flat, B, K) # [B*K, C_hidden]
            
            x_flat_bk_h = x_att_flat + x_residual_flat # Add residual
            x_flat_bk_h = F.silu(x_flat_bk_h, inplace=True)
            
        # Reshape back to [B, K, C_hidden]
        x_out_b_k_c = x_flat_bk_h.view(B, K, self.hidden_dim)
        return x_out_b_k_c