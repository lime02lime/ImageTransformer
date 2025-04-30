import matplotlib.pyplot as plt


def visualise_patched_input(result, label_str, patch_size):
    fig, axs = plt.subplots(14, 14)
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.imshow(result[idx].reshape(patch_size, patch_size))
        ax.axis('off')
    plt.suptitle(f'Patched input\n Label: {label_str}')
    plt.show()
    return    
