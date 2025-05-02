import matplotlib.pyplot as plt


def visualise_patched_input(result, label_str, patch_size, ):
    N_patches, _ = result.shape
    sqrt_N_patches = int(N_patches ** 0.5)
    fig, axs = plt.subplots(sqrt_N_patches, sqrt_N_patches, figsize=(8, 8))
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.imshow(result[idx].reshape(patch_size, patch_size))
        ax.axis('off')
    plt.suptitle(f'Patched input\n Label: {label_str}')
    plt.show()
    return    
