import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

from collections import defaultdict

def compute_accuracy_t2i(output, target, topk=(1,)):
    # Shape: (B, 4C, C+1)
    B, fourC, C_plus_1 = output.shape    
    C = fourC // 4
    numClasses = C_plus_1 - 1

    # Make two copies of the output
    output_log_softmax = F.log_softmax(output.clone(), dim=2)  # log probs version
    output_target = output.clone()  # copy to modify

    # Modify output_target[i, j, 0] = output[i, j, (j % C) + 1]
    for i in range(B):
        for j in range(fourC):
            idx = (j % numClasses) + 1
            output_target[i, j, 0] = output_target[i, j, idx]

    # --- KL Divergence ---
    output_target_softmax = F.softmax(output_target, dim=2)  # convert target to probabilities

    output_softmax = F.softmax(output.clone(), dim=2) 
    # print("output first vector\n", output[0,0,:])
    # print("output_target first vector\n", output[0,0,:])

    # print("output_softmax first vector\n", output_softmax[0,0,:])
    # print("output_target_softmax first vector\n", output_target_softmax[0,0,:])

    # print("output_log_softmax shape: ", output_log_softmax.shape)
    # print("output_target_softmax shape: ", output_target_softmax.shape)

    # Compute KL divergence using F.kl_div
    kl_div = F.kl_div(
        output_log_softmax,
        output_target_softmax,
        reduction='none',  # keep shape: (B, 4C, C+1)
        log_target=False
    )

    # print("kl_div shape: ", kl_div.shape)

    # Sum over class dimension to get (B, 4C)
    kl_div_per_row = kl_div.sum(dim=2)  # shape: (B, 4C)

    # print("kl_div_per_row shape: ", kl_div_per_row.shape)
    

    # --- Accuracy computation based on kl_div_per_row (smaller is better) ---
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top-k predictions from kl_div_per_row
    topk_vals, topk_indices = kl_div_per_row.topk(maxk, dim=1, largest=False, sorted=True)

    
    topk_indices = topk_indices % numClasses

    # print("topk_vals shape: ", topk_vals.shape)
    # print("topk_indices shape: ", topk_indices.shape)

    pred = topk_indices.t()  # shape: (maxk, batch_size)

    # print("pred\n", pred)
    # print("target\n", target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # input("press enter to continue")

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res



def compute_accuracy_i2t(output, target, topk=(1,), printoutput=False):
    """Computes the accuracy over the k top predictions for
    the specified values of k, and prints the top-k values and indices.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    

    # Set print options to show full tensor with 1 decimal point precision
    # torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=200, precision=1)
    torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=1000, precision=1)

    logits_i2t=output
    # logits_t2i=output.t()
    logits_per_image=output

    logits_per_text = logits_i2t.T  # Shape: [37, 32]
    # Step 2: Build text → list of image indices mapping
    text_to_image_indices = defaultdict(list)
    for img_idx, text_idx in enumerate(target):
        text_to_image_indices[int(text_idx)].append(img_idx)

    # Step 3: Create soft target matrix [num_texts, num_images]
    targets_T = torch.zeros_like(logits_per_text)  # [num_texts, num_images]
    for text_idx, image_list in text_to_image_indices.items():
        if len(image_list) > 0:
            weight = 1.0 / len(image_list)
            targets_T[text_idx, image_list] = weight

    # Step 4: Filter rows for texts with atleast one matching image
    valid_mask = targets_T.sum(dim=1) > 0  
    filtered_logits_per_text = logits_per_text[valid_mask]
    filtered_targets = targets_T[valid_mask]
    filtered_target_indices = filtered_targets.argmax(dim=1)

    # Step 5: Compute soft cross-entropy using KL divergence
    log_probs = F.log_softmax(filtered_logits_per_text, dim=1)   # log p_model
    loss_t2i = F.kl_div(log_probs, filtered_targets, reduction='batchmean')

    #############################################################
            
            
    # Image to text loss
    #############################################################
            
    
    # Step 1: Build a mask for images whose label indices is in filtered_target_indices
    valid_image_indices = set(filtered_target_indices.tolist())  # Convert to set for faster lookup

    # Now build the mask by checking if the index is in valid_image_indices
    image_mask = torch.tensor([i in valid_image_indices for i in range(len(target))])

    # Step 2: Filter logits and labels
    filtered_logits_per_image = logits_per_image[image_mask]
    filtered_labels = target[image_mask]

    # Step 3: Compute cross entropy loss
    # loss_i2t = F.cross_entropy(filtered_logits_per_image, filtered_labels)
    # loss_i2t = F.cross_entropy(logits_per_image, label)    

    # probs_images = F.softmax(filtered_logits_per_image, dim=1)   # log p_model  
    probs_images = F.softmax(logits_per_image, dim=1)   # log p_model  
    image_no_mask = torch.ones(len(target), dtype=torch.bool)

    log_probs_images = F.log_softmax(filtered_logits_per_image, dim=1)   # log p_model
    num_classes = log_probs_images.size(1)
    filtered_labels_onehot = F.one_hot(filtered_labels, num_classes=num_classes).float()
    loss_i2t = F.kl_div(log_probs_images, filtered_labels_onehot, reduction='batchmean')    
  

    num_filtered_logits_per_image = filtered_logits_per_image.shape[0]
    num_filtered_logits_per_text = filtered_logits_per_text.shape[0]

    
    print("num_filtered_logits_per_image_compute_accuracy = ", num_filtered_logits_per_image)
    print("num_filtered_logits_per_text_compute_accuracy = ", num_filtered_logits_per_text)


    # loss_i2t = F.cross_entropy(logits_i2t, target)
    # loss_t2i = F.cross_entropy(logits_t2i, target)
    

    if printoutput:
        # print("output_logits\n", output)
        # print("output_target\n", target)
        print("loss_i2t: ", loss_i2t.item())
        print("loss_t2i: ", loss_t2i.item())
        # input("Press enter1 to continue")

    torch.set_printoptions(profile='default')

    if isinstance(output, (tuple, list)):
        output = output[0]

    # Get top-k predictions
    topk_vals, topk_indices = output.topk(maxk, dim=1, largest=True, sorted=True)

    topk_vals_test, topk_indices_test = output.topk(10, dim=1, largest=True, sorted=True)    

    # print("topk_vals_i2t\n", topk_vals_test)
    # print("topk_indices_i2t\n", topk_indices_test)
    # print("target_i2t\n")
    # print(target)

    pred = topk_indices.t()  # shape: (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res, loss_i2t, loss_t2i

def compute_accuracy(output, target, topk=(1, ), printoutput=False):
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

    if printoutput:
        print("output_logits\n", output)
        print("output_target\n", target)
        input("Press enter1 to continue")


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
