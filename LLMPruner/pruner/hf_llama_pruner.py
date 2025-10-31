import torch
import torch.nn as nn

import LLMPruner.torch_pruning as tp
from LLMPruner.torch_pruning import BasePruningFunc, ops

from copy import deepcopy
import random
from functools import reduce
from operator import mul

from typing import Callable, Sequence, Tuple, Dict
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

##############################
# Pruners
##############################

class HFRMSNormPrunner(BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        #print("Pruning RMSNorm Layer: {}".format(layer))
        keep_idxs = list(set(range(layer.weight.size(0))) - set(idxs))
        keep_idxs.sort()
        
        layer.weight = torch.nn.Parameter(
            layer.weight[keep_idxs]
        )
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.size(0)

    def get_in_channels(self, layer):
        return layer.weight.size(0)

class HFAttentionPrunner(BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        assert len(idxs) % layer.num_heads == 0
        #print("Prune IDX in HFAttentionPruner: ", idxs)
        for sub_layer in [layer.o_proj]:
            keep_idxs = list(set(range(sub_layer.out_features)) - set(idxs))
            keep_idxs.sort()
            sub_layer.out_features = sub_layer.out_features-len(idxs)

            sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[keep_idxs])
            if sub_layer.bias is not None:
                sub_layer.bias = torch.nn.Parameter(sub_layer.bias.data[keep_idxs])

        for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:  
            keep_idxs = list(set(range(sub_layer.in_features)) - set(idxs))
            keep_idxs.sort()
            sub_layer.in_features = sub_layer.in_features-len(idxs)
            sub_layer.weight = torch.nn.Parameter(
                sub_layer.weight.data[:, keep_idxs]
            )

        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.hidden_size

    def get_in_channels(self, layer):
        return layer.hidden_size
    

class HFLinearPrunner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()
        idxs.sort()
        layer.out_features = layer.out_features-len(idxs)

        keep_weight = layer.weight.data[keep_idxs]
        remove_weight = layer.weight.data[idxs]

        sim = torch.mm(remove_weight, keep_weight.t())
        max_indices = torch.argmax(sim, dim=-1)
        keep_weight[max_indices] += remove_weight
        cnt = torch.ones((keep_weight.size(0), 1), device=keep_weight.device)
        cnt[torch.max(sim, dim=-1).indices] += 1
        keep_weight = keep_weight / cnt

        layer.weight = torch.nn.Parameter(keep_weight)
        if layer.bias is not None:
            keep_bias = layer.bias.data[keep_idxs]
            remove_bias = layer.bias.data[idxs]
            keep_bias[max_indices] += remove_bias
            keep_bias = keep_bias / cnt
            layer.bias = torch.nn.Parameter(keep_bias)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()
        layer.in_features = layer.in_features-len(idxs)

        keep_weight = layer.weight.data[:, keep_idxs]
        remove_weight = layer.weight.data[:, idxs]

        sim = torch.mm(remove_weight.t(), keep_weight)
        max_indices = torch.argmax(sim, dim=-1)
        keep_weight[:, max_indices] += remove_weight
        cnt = torch.ones((1, keep_weight.size(1)), device=keep_weight.device)
        cnt[:, torch.max(sim, dim=-1).indices] += 1
        #keep_weight = keep_weight / cnt

        layer.weight = torch.nn.Parameter(keep_weight)
        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features

hf_attention_pruner = HFAttentionPrunner()
hf_rmsnorm_pruner = HFRMSNormPrunner()
hf_linear_pruner = HFLinearPrunner()

##############################
# Importance
##############################
class MagnitudeImportance(tp.importance.Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
    
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                group_imp.append(local_norm)
            # Linear in_channels
            elif prune_fn in [
                tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels
            ]:    
                w = layer.weight
                local_norm = w.abs().pow(self.p).sum(0)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
            # RMSNorm
            elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
                # regularize BN
                w = layer.weight.data[idxs]
                local_norm = w.abs().pow(self.p)
                group_imp.append(local_norm)
            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                w = layer.weight.data[:, idxs]
                local_norm = w.abs().pow(self.p)
                group_imp.append(local_norm)
            # Attention
            elif prune_fn == hf_attention_pruner.prune_out_channels:
                local_norm = 0
                for sub_layer in [layer.o_proj]:
                    w_out = sub_layer.weight.data[idxs]
                    local_norm += w_out.abs().pow(self.p).sum(1)

                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
                    w_in = sub_layer.weight.data[:, idxs]
                    local_norm += w_in.abs().pow(self.p).sum(0)
                group_imp.append(local_norm)
            print("####groupi:",i)  
            i = i + 1     

        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp

class FusionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)  # 输出一个标量作为权重

    def forward(self, fine_grained_grad, coarse_grained_grad):
       
        # 将两种信息拼接在一起
        combined_input = torch.cat((fine_grained_grad, coarse_grained_grad), dim=1) #dim=1
        
        # 通过全连接层进行权重学习
        hidden = self.fc1(combined_input)
        hidden = self.relu(hidden)
        weight = torch.sigmoid(self.output_layer(hidden))  # 使用 sigmoid 激活函数确保权重在 0 和 1 之间
        
        print("fusion ratio and shape:",weight, weight.shape)
        # 将权重应用到两种信息上
        
        fused_output = weight * fine_grained_grad + (1 - weight) * coarse_grained_grad
        print("fused_output",fused_output.shape)
        
        return fused_output
        

class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor
        self.fusion_model = None #FusionModel(input_size * 2) 

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction=='second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    '''
    def _fuse(self, fine_grained_grad, coarse_grained_grad):
        #fine_grained_grad = salience.abs().sum(1)
        #coarse_grained_grad= salience.sum(1).abs()
        epsilon = 1e-8
        fine_grained_norm = torch.norm(fine_grained_grad ) + epsilon
        coarse_grained_norm = torch.norm(coarse_grained_grad) + epsilon
        normalized_fine_grained_grad = fine_grained_grad / fine_grained_norm
        normalized_coarse_grained_grad = coarse_grained_grad / coarse_grained_norm
        fusion_factor = normalized_coarse_grained_grad.norm() / (normalized_fine_grained_grad.norm() + normalized_coarse_grained_grad.norm()) 
        print("####fusion_factor:",fusion_factor)
        local_norm = (1 - fusion_factor) * fine_grained_grad + fusion_factor * coarse_grained_grad
        return local_norm
    
    '''
    def _fuse(self, fine_grained_grad, coarse_grained_grad):
        
            # 每次调用时都重新初始化 fusion_model
            print("fine_grained_grad.size",fine_grained_grad.size,coarse_grained_grad.size)
            input_size = fine_grained_grad.size(1) + coarse_grained_grad.size(1)
            self.fusion_model = FusionModel(input_size)
            return self.fusion_model(fine_grained_grad, coarse_grained_grad)
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
    
        print("hf_llama_pruner.py TaylorImportance self.taylor#######",self.taylor)#jliu
        group_imp = []
        i = 0
        for dep, idxs in group:
            
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn not in [
                tp.prune_linear_out_channels, tp.prune_linear_in_channels, 
                hf_rmsnorm_pruner.prune_out_channels, tp.prune_embedding_out_channels, hf_attention_pruner.prune_out_channels,
                hf_linear_pruner.prune_out_channels, hf_linear_pruner.prune_in_channels
            ]:
                continue
            print("####in dependency sub-group No.",i) 
            if prune_fn in [hf_attention_pruner.prune_out_channels]:
                salience = {}
                for sub_layer in [layer.o_proj, layer.q_proj, layer.k_proj, layer.v_proj]:
                    salience[sub_layer] = sub_layer.weight * sub_layer.weight.grad
                    print("####in hf_attention_pruner.prune_out_channels sub-layer ", sub_layer) 
                    if self.taylor in ['param_second']:
                        salience[sub_layer] = sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
                    elif self.taylor in ['param_mix']: 
                        salience[sub_layer] = -salience + 0.5 * sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight   
                    elif self.taylor in ['sec_fuse']: 
                        coarse_grained_grad = sub_layer.weight * sub_layer.weight.grad
                        fine_grained_grad = sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
                        salience[sub_layer]  = self._fuse(fine_grained_grad,  coarse_grained_grad)



                    
            else:
                print("####in dependency NOT in hf_attention_pruner.prune_out_channels ") 
                salience = layer.weight * layer.weight.grad

                if self.taylor in ['param_second']:
                    salience = layer.weight * layer.weight.acc_grad * layer.weight
                elif self.taylor in ['param_mix']: 
                    salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight
                elif self.taylor in ['second_fuse','mix_fuse']: 
                     coarse_grained_grad = sub_layer.weight * sub_layer.weight.grad
                     fine_grained_grad = sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
                     salience  = self._fuse(fine_grained_grad,  coarse_grained_grad)


            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
                if self.taylor == 'vectorize':
                    local_norm = salience.sum(1).abs()
                    '''
                    coarse_grained_grad = salience.sum(1).abs().unsqueeze(1)  # Keep batch dimensions
                    fine_grained_grad = salience.abs().sum(1).unsqueeze(1)  # Keep batch dimensions
                    print("&&&&#fine_grained_grad shape",fine_grained_grad.shape)
                    print("&&&&&coarse_grained_grad shape",coarse_grained_grad.shape)
                    
                    local_norm  = self._fuse( fine_grained_grad, coarse_grained_grad)
                    '''
                elif 'param' in self.taylor:
                   local_norm = salience.abs().sum(1)
                elif 'fuse'  in self.taylor:
                    print("Importance Linear out_channels")
                    #local_norm = self._fuse(salience.abs().sum(1), salience.sum(1).abs())
                    
                    #print("salience.sum(1).abs() shape",salience.sum(1).abs().shape)
                    #print("salience.abs().sum(1) shape",salience.abs().sum(1).shape)
                    coarse_grained_grad = salience.sum(1).abs().unsqueeze(1)  # Keep batch dimensions
                    fine_grained_grad = salience.abs().sum(1).unsqueeze(1)  # Keep batch dimensions
                    #print("VVVfine_grained_grad shape",fine_grained_grad.shape)
                    #print("VVVcoarse_grained_grad shape",coarse_grained_grad.shape)
                    
                    fused_output  = self._fuse(fine_grained_grad,  coarse_grained_grad)
                    #print("fused_output shape",fused_output.shape)
                    local_norm = fused_output.squeeze(1)
                    
                    #print("localnorm shape",local_norm.shape)

                else:
                    raise NotImplementedError
                group_imp.append(local_norm)

            # Linear in_channels
            elif prune_fn in [tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels]:
                print("Importance Linear in_channels")
                if self.taylor == 'vectorize':
                    local_norm = salience.sum(0).abs()
                    '''
                    coarse_grained_grad = salience.sum(0).abs().unsqueeze(1)  # Keep batch dimensions
                    fine_grained_grad = salience.abs().sum(0).unsqueeze(1)  # Keep batch dimensions
                    print("&&&&#fine_grained_grad shape",fine_grained_grad.shape)
                    print("&&&&&coarse_grained_grad shape",coarse_grained_grad.shape)
                    
                    local_norm  = self._fuse( fine_grained_grad, coarse_grained_grad)
                    '''
                elif 'param' in self.taylor:
                    local_norm = salience.abs().sum(0)
                elif 'fuse'  in self.taylor:
                    #local_norm = self._fuse(salience.abs().sum(0), salience.sum(0).abs())
                    
                    #print("######salience.sum(1).abs() shape",salience.sum(1).abs().shape)
                    #print("######salience.abs().sum(0) shape",salience.abs().sum(0).shape)
                    
                    coarse_grained_grad = salience.sum(0).abs().unsqueeze(1)  # Keep batch dimensions
                    fine_grained_grad = salience.abs().sum(0).unsqueeze(1)  # Keep batch dimensions
                    print("&&&&#fine_grained_grad shape",fine_grained_grad.shape)
                    print("&&&&&coarse_grained_grad shape",coarse_grained_grad.shape)
                    
                    fused_output  = self._fuse( fine_grained_grad, coarse_grained_grad)
                    print("&&&&&fused_output shape",fused_output.shape)
                    local_norm = fused_output.squeeze(1)
                    
                    #print("&&&&&&localnorm shape",local_norm.shape)
                else:
                    raise NotImplementedError
                print("idxs.max():",max(idxs))
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # RMSNorm
            elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
                local_norm = salience.abs()
                group_imp.append(local_norm)

            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                print("Importance embedding_out_channels")
                if self.taylor == 'vectorize':
                    local_norm = salience[:, idxs].sum(0).abs()
                elif 'param' in self.taylor:
                    local_norm = salience[:, idxs].abs().sum(0)
                elif 'fuse'  in self.taylor:
                    #local_norm = self._fuse(salience.abs().sum(0), salience.sum(0).abs())
                    fine_grained_grad = salience[:, idxs].sum(0).abs().unsqueeze(1)  # Keep batch dimensions
                    coarse_grained_grad = salience[:, idxs].abs().sum(0).unsqueeze(1)  # Keep batch dimensions
                    fused_output  = self._fuse(fine_grained_grad, coarse_grained_grad)
                    local_norm = fused_output.squeeze(1)
                else:
                    raise NotImplementedError
                group_imp.append(local_norm)

            # Attention
            elif prune_fn == hf_attention_pruner.prune_out_channels:
                print("Importance hf_attention_pruner.prune_out_channels")
                local_norm = 0
                for sub_layer in [layer.o_proj]: #linear out channel, first dim in linear.weight
                    if self.taylor == 'vectorize':
                        local_norm += salience[sub_layer].sum(1).abs()
                    elif 'param' in self.taylor: 
                        local_norm += salience[sub_layer].abs().sum(1)   
                    elif 'fuse'  in self.taylor:
                        print("fuse attention out channel !!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        #local_norm += self._fuse(salience.abs().sum(1), salience.sum(1).abs())
                        fine_grained_grad = salience[sub_layer].sum(1).abs().unsqueeze(1)  # Keep batch dimensions
                        coarse_grained_grad = salience[sub_layer].abs().unsqueeze(1)  # Keep batch dimensions
                        fused_output  = self._fuse(fine_grained_grad, coarse_grained_grad)
                        local_norm += fused_output.squeeze(1)
                    else:
                        raise NotImplementedError                
                
                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]: # linear in channel, second dim in linear.weight
                    if self.taylor == 'vectorize':
                        local_norm += salience[sub_layer].sum(0).abs() 
                    elif 'param' in self.taylor == 'param':
                        local_norm += salience[sub_layer].abs().sum(0)
                    elif 'fuse'  in self.taylor:
                        print("fuse attention in channel !!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        #local_norm += self._fuse(salience.abs().sum(0), salience.sum(0).abs())
                        fine_grained_grad = salience[sub_layer].sum(0).abs().unsqueeze(1)  # Keep batch dimensions
                        coarse_grained_grad = salience[sub_layer].abs().sum(0).unsqueeze(1)  # Keep batch dimensions
                        fused_output  = self._fuse(fine_grained_grad, coarse_grained_grad)
                        local_norm += fused_output.squeeze(1) 
                    else:
                        raise NotImplementedError
                group_imp.append(local_norm)
             
            i = i + 1   

        if len(group_imp)==0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp
