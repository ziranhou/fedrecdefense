import torch
import torch.nn as nn
from parse import args

import torch
import torch.nn as nn
from parse import args
import defense  # Assuming this import is required for defense functions


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, layers):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i - 1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def normbound(self, items_emb_grad):
        # Apply norm-bound defense
        norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
        if len(norm.shape) == 1:  # bias
            too_large = norm[0] > args.grad_limit
        else:  # weights
            too_large = norm[:, 0] > args.grad_limit
        items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit)
        return items_emb_grad

    def train_(self, clients, batch_clients_idx, epoch, mal_start_ind):
        items_emb = self.items_emb.weight
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]

        for idx in batch_clients_idx:
            client = clients[idx]
            # Ensure epoch is passed to the client's train_ method
            items, items_emb_grad, linear_layers_grad, loss = client.train_(items_emb, linear_layers, epoch)

            with torch.no_grad():
                # Apply norm-bound defense if specified
                if args.defense == 'NormBound':
                    items_emb_grad = self.normbound(items_emb_grad)

                # Update gradients
                batch_items_emb_grad[items] += items_emb_grad
                for i in range(len(linear_layers)):
                    if args.defense == 'NormBound':
                        if linear_layers_grad is not None:
                            linear_layers_grad[i][0] = self.normbound(linear_layers_grad[i][0])
                            linear_layers_grad[i][1] = self.normbound(linear_layers_grad[i][1])

                    if linear_layers_grad is not None:
                        batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                        batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            # Update items_emb and linear layers with gradients
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
            for i in range(len(linear_layers)):
                self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)

        return batch_loss

    def eval_(self, clients):
        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results = 0, 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(items_emb, linear_layers)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result

        return test_results / test_cnt, target_results / target_cnt


# class FedRecServer(nn.Module):
#     def __init__(self, m_item, dim, layers):
#         super().__init__()
#         self.m_item = m_item
#         self.dim = dim
#         self.layers = layers
#
#         self.items_emb = nn.Embedding(m_item, dim)
#         nn.init.normal_(self.items_emb.weight, std=0.01)
#
#         layers_dim = [2 * dim] + layers + [1]
#         self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i-1], layers_dim[i])
#                                             for i in range(1, len(layers_dim))])
#         for layer in self.linear_layers:
#             nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#             nn.init.zeros_(layer.bias)
#
#     def train_(self, clients, batch_clients_idx, epoch,mal_start_ind):
#         items_emb = self.items_emb.weight
#         linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
#         batch_loss = []
#         batch_items_emb_grad = torch.zeros_like(items_emb)
#         batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]
#
#         for idx in batch_clients_idx:
#             client = clients[idx]
#             items, items_emb_grad, linear_layers_grad, loss = client.train_(items_emb, linear_layers)
#
#             with torch.no_grad():
#                 batch_items_emb_grad[items] += items_emb_grad
#                 for i in range(len(linear_layers)):
#                     batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
#                     batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]
#
#             if loss is not None:
#                 batch_loss.append(loss)
#
#         with torch.no_grad():
#             self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
#             for i in range(len(linear_layers)):
#                 self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
#                 self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)
#         return batch_loss
#
#     def eval_(self, clients):
#         items_emb = self.items_emb.weight
#         linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
#         test_cnt, test_results = 0, 0.
#         target_cnt, target_results = 0, 0.
#
#         with torch.no_grad():
#             for client in clients:
#                 test_result, target_result = client.eval_(items_emb, linear_layers)
#                 if test_result is not None:
#                     test_cnt += 1
#                     test_results += test_result
#                 if target_result is not None:
#                     target_cnt += 1
#                     target_results += target_result
#         return test_results / test_cnt, target_results / target_cnt

import defense
class PopServer(nn.Module):
    def __init__(self, m_item, dim, layers):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i-1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            
    def normbound(self, items_emb_grad):
        norm = items_emb_grad.norm(2, dim=-1, keepdim=True) 
        if len(norm.shape) == 1: # bias
            too_large = norm[0] > args.grad_limit
        else: # weights
            too_large = norm[:,0] > args.grad_limit
        items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit) 
        return items_emb_grad

    def compute_update_difference(self, update1, update2):
        """
        计算两个更新之间的欧几里得距离
        """
        return torch.norm(update1 - update2, p=2)

    def weighted_median(self, client_updates, client_losses):
        """
        计算加权中位数聚合，考虑更新之间的差异。
        根据客户端更新与其他客户端更新的差异来计算加权。
        """
        # 计算每个客户端更新之间的差异
        distances = []
        for i in range(len(client_updates)):
            diff = 0
            for j in range(len(client_updates)):
                if i != j:
                    diff += self.compute_update_difference(client_updates[i], client_updates[j])
            distances.append(diff)

        # 计算权重：差异越小，权重越大
        max_distance = max(distances) if distances else 1
        weights = [(max_distance - dist) / max_distance for dist in distances]

        # 排序客户端更新
        sorted_updates = sorted(zip(weights, client_updates), key=lambda x: x[0], reverse=True)
        sorted_weights = [s[0] for s in sorted_updates]
        sorted_updates = [s[1] for s in sorted_updates]

        # 加权中位数聚合
        total_weight = sum(sorted_weights)
        half_weight = total_weight / 2
        cumulative_weight = 0

        for weight, update in zip(sorted_weights, sorted_updates):
            cumulative_weight += weight
            if cumulative_weight >= half_weight:
                return update
        return sorted_updates[0]

    def train_(self, clients, batch_clients_idx,epoch,mal_start_ind):
        items_emb = self.items_emb.weight
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]

        batch_items_inter = torch.zeros([len(items_emb), 1])
        all_losses = []
        client_updates = []


        batch_items_inter = torch.zeros([len(items_emb),1])
        if args.defense != 'NoDefense' and args.defense != 'NormBound' and args.defense[:6] != 'Regula' and args.defense != 'Ours':
            batch_items = [[] for i in range(len(batch_clients_idx))]
            batch_items_grads = torch.zeros((len(batch_clients_idx), len(self.items_emb.weight),self.dim)).to(args.device)
            batch_linear_grads = [[[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers] for i in range(len(batch_clients_idx))]
        
        for idx, user in enumerate(batch_clients_idx):
            client = clients[user]
            items, items_emb_grad, linear_layers_grad, loss = client.train_(items_emb,linear_layers,epoch)

            with torch.no_grad():
                if args.defense == 'NormBound': 
                    items_emb_grad = self.normbound(items_emb_grad)
                if args.defense != 'NoDefense' and args.defense != 'NormBound' and args.defense[:6] != 'Regula' and args.defense != 'Ours':
                    batch_items_grads[idx,items,:] = items_emb_grad
                    if isinstance(items,list):
                        batch_items[idx] = items
                    else:
                        batch_items[idx] = items.cpu().numpy().tolist()

                batch_items_emb_grad[items] += items_emb_grad
                # batch_items_inter[items] += 1

                # 记录损失
                all_losses.append(loss)

                # 将 [len(items), dim] 大小的梯度扩展到 [m_item, dim]
                full_update = torch.zeros_like(self.items_emb.weight)  # [m_item, dim]
                full_update[items] = items_emb_grad  # 把对应索引的更新写入full_update
                client_updates.append(full_update)

                batch_items_emb_grad[items] += items_emb_grad

                
                for i in range(len(linear_layers)):
                    if args.defense == 'NormBound':
                        if linear_layers_grad is None:
                            continue
                        linear_layers_grad[i][0] = self.normbound(linear_layers_grad[i][0])
                        linear_layers_grad[i][1] = self.normbound(linear_layers_grad[i][1])

                    if args.defense != 'NoDefense' and args.defense != 'NormBound' and args.defense[:6] != 'Regula' and args.defense != 'Ours':
                        if linear_layers_grad is None and i == 0:
                            batch_linear_grads[idx] = None
                            continue
                        if linear_layers_grad is not None:
                            batch_linear_grads[idx][i][0] = linear_layers_grad[i][0].cpu()
                            batch_linear_grads[idx][i][1] = linear_layers_grad[i][1].cpu()
                        # batch_linear_grads[idx][i][0] = linear_layers_grad[i][0].cpu()
                        # batch_linear_grads[idx][i][1] = linear_layers_grad[i][1].cpu()

                    if linear_layers_grad != None:
                        batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                        batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            if args.defense == 'Ours':
                # 1. 防止输入中出现不合理的零值
                batch_items_inter[batch_items_inter == 0] = 1

                # 2. 计算输入的扰动并进行修正
                noise = torch.randn_like(batch_items_inter) * 0.05
                batch_items_inter = batch_items_inter + noise

                # 3. 进行模型参数更新（梯度下降+L2正则化）
                self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
                self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)

                # 4. 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)



                # 使用加权中位数聚合
                # valid_pairs = [(update, loss) for update, loss in zip(client_updates, all_losses) if loss is not None]
                # if valid_pairs:
                #     filtered_updates, filtered_losses = zip(*valid_pairs)
                #     robust_update = self.weighted_median(filtered_updates, filtered_losses)
                #     # 更新全局物品嵌入
                #     self.items_emb.weight.data.add_(robust_update, alpha=-args.lr)

                # 使用加权中位数聚合
                # valid_pairs = [(update, loss) for update, loss in zip(client_updates, all_losses) if loss is not None]
                # if valid_pairs:
                #     filtered_updates, filtered_losses = zip(*valid_pairs)  # 解压有效的更新和损失
                #     robust_update = self.weighted_median(filtered_updates, filtered_losses)
                #     # 更新全局物品嵌入
                #     self.items_emb.weight.data.add_(robust_update, alpha=-args.lr)
                #
                #     # 对线性层进行更新
                #     for i in range(len(self.linear_layers)):
                #         # 更新线性层的权重
                #         self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                #         # 更新线性层的偏置
                #         self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)



            elif args.defense == 'NoDefense' or args.defense == 'NormBound' or args.defense[:6] == 'Regula':
                batch_items_inter[batch_items_inter == 0] = 1
                self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
                for i in range(len(linear_layers)):
                    self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                    self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)
                
            else:
                import numpy as np
                batch_current_grads = torch.zeros_like(items_emb)
                for i in range(batch_items_grads.shape[1]):
                    user_idx = [i in x for x in batch_items]
                    if sum(user_idx) == 0:
                        batch_current_grads[i] = torch.zeros_like(items_emb[0]).to(args.device)
                    else:
                        before_defense_grads = batch_items_grads[user_idx,i,:].cpu()
                        corrupted_count=int(sum(user_idx)*args.clients_limit)
                        current_grads = defense.defend[args.defense](np.array(before_defense_grads), sum(user_idx), corrupted_count) 
                        batch_current_grads[i] = torch.from_numpy(current_grads).to(args.device)
                    
                self.items_emb.weight.data.add_(batch_current_grads, alpha=-args.lr)

                for i in range(len(linear_layers)):
                    pending_weight, pending_bias = [], []
                    for x in batch_linear_grads:
                        if x is None:
                            continue
                        pending_weight.append(x[i][0])
                        pending_bias.append(x[i][1])
                    for j in range(len(pending_weight)):
                        pending_weight[j] = np.array(pending_weight[j]).reshape(-1)
                        pending_bias[j] = np.array(pending_bias[j]).reshape(-1)

                    corrupted_count = int(len(batch_clients_idx)*args.clients_limit)
                    current_weight_grad = defense.defend[args.defense](np.array(pending_weight), len(batch_clients_idx), corrupted_count)
                    current_weight_grad = torch.from_numpy(current_weight_grad.reshape(len(self.linear_layers[i].weight),-1)).to(args.device)

                    current_bias_grad = defense.defend[args.defense](np.array(pending_bias), len(batch_clients_idx), corrupted_count)
                    current_bias_grad = torch.from_numpy(current_bias_grad).to(args.device)
                    self.linear_layers[i].weight.data.add_(current_weight_grad, alpha=-args.lr)
                    self.linear_layers[i].bias.data.add_(current_bias_grad, alpha=-args.lr)  
        return batch_loss

    def eval_(self, clients):
        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results = 0, 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(items_emb, linear_layers)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result
        return test_results / test_cnt, target_results / target_cnt
