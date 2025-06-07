import torch
import torch.nn.functional as F

from collections import defaultdict


def compute_accuracy_with_i2t_t2i_loss(output, target, topk=(1,), printoutput=False):

    """Computes the accuracy over the k top predictions for
    the specified values of k. Additionally, computes the Image-to-Text (I2T) and
    Text-to-Image (T2I) losses.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k, Image-to-text (I2T) loss, and Text-to-Image (T2I) loss.
    """
    
    maxk = max(topk)
    batch_size = target.size(0)    

    # Set print options to show full tensor with 1 decimal point precision
    # torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=200, precision=1)
    torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=1000, precision=1)

    # Image to Text loss
    #############################################################
    logits_i2t=output    
    logits_per_image=output
    loss_i2t = F.cross_entropy(logits_i2t, target)

    # Text to Image loss
    #############################################################

    logits_per_text = logits_i2t.T 
    # Build text → list of image indices mapping
    text_to_image_indices = defaultdict(list)
    for img_idx, text_idx in enumerate(target):
        text_to_image_indices[int(text_idx)].append(img_idx)

    # Create soft target matrix [num_texts, num_images]
    targets_T = torch.zeros_like(logits_per_text)  # [num_texts, num_images]
    for text_idx, image_list in text_to_image_indices.items():
        if len(image_list) > 0:
            weight = 1.0 / len(image_list)
            targets_T[text_idx, image_list] = weight

    # Filter rows for texts with atleast one matching image
    valid_mask = targets_T.sum(dim=1) > 0  
    filtered_logits_per_text = logits_per_text[valid_mask]
    filtered_targets = targets_T[valid_mask]
    filtered_target_indices = filtered_targets.argmax(dim=1)

    # Compute soft cross-entropy using KL divergence
    log_probs = F.log_softmax(filtered_logits_per_text, dim=1)   # log p_model
    loss_t2i = F.kl_div(log_probs, filtered_targets, reduction='batchmean')    

    torch.set_printoptions(profile='default')

    # Accuracy calculation
    #############################################################

    if isinstance(output, (tuple, list)):
        output = output[0]

    # Get top-k predictions
    topk_vals, topk_indices = output.topk(maxk, dim=1, largest=True, sorted=True)

    topk_vals_test, topk_indices_test = output.topk(10, dim=1, largest=True, sorted=True)        

    pred = topk_indices.t()  # shape: (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res, loss_i2t, loss_t2i

def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res
