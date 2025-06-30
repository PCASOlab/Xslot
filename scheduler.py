def cyclic_learning_rate(current_epoch, max_lr, min_lr, cycle_length=10):
    """
    Function to compute the learning rate based on a cyclic schedule.
    
    Parameters:
        current_epoch (int): The current epoch number.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.
        cycle_length (int): The length of the cycle in epochs.
        
    Returns:
        float: The computed learning rate for the current epoch.
    """
    cycle_position = current_epoch % cycle_length
    
    if cycle_position < cycle_length / 2:
        # Linearly decay the learning rate from max_lr to min_lr in the first half of the cycle
        slope = (max_lr - min_lr) / (cycle_length / 2)
        lr = max_lr - slope * cycle_position
    else:
        # Linearly increase the learning rate from min_lr to max_lr in the second half of the cycle
        slope = (max_lr - min_lr) / (cycle_length / 2)
        lr = min_lr + slope * (cycle_position - cycle_length / 2)
    
    return lr