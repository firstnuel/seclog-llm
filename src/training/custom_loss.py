import torch.nn.functional as F

def sparse_alignment_loss(pred_embeddings, target_embeddings, alphas, lambda_l1=0.01):
    # 1. Alignment: Make projected log embedding match target concept
    mse_loss = F.mse_loss(pred_embeddings, target_embeddings)
    
    # 2. Sparsity: Force alphas to be 0 unless necessary
    l1_loss = alphas.abs().mean()
    
    return mse_loss + (lambda_l1 * l1_loss)
