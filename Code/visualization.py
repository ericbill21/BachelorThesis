import matplotlib.pyplot as plt
from datetime import datetime

# Disable interactive mode for matplotlib
plt.ioff()

def plot_loss_and_accuracy(dataset_name, all_train_losses, all_train_accuracies,all_val_losses, all_val_accuracies):
    model_names = list(all_train_losses.keys())

    fig, axs = plt.subplots(2, 2, figsize=(24, 18))

    fig.suptitle(f'Loss and accuracy for {dataset_name}')

    axs[0, 0].set_title('Training loss')
    axs[0, 1].set_title('Validation loss')
    axs[1, 0].set_title('Training accuracy')
    axs[1, 1].set_title('Validation accuracy')

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 1].set_xlabel('Epoch')

    axs[0, 0].set_ylabel('Loss')
    axs[0, 1].set_ylabel('Loss')
    axs[1, 0].set_ylabel('Accuracy[%]')
    axs[1, 1].set_ylabel('Accuracy[%]')

    axs[0, 0].grid(True)
    axs[0, 1].grid(True)
    axs[1, 0].grid(True)
    axs[1, 1].grid(True)

    color_map = {}    
    colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    
    for model_name in model_names:
        color_map[model_name] = next(colors)

        axs[0, 0].plot(all_train_losses[model_name].mean(dim=1), label=model_name, c=color_map[model_name])
        axs[0, 1].plot(all_val_losses[model_name].mean(dim=1), label=model_name, c=color_map[model_name])
        axs[1, 0].plot(all_train_accuracies[model_name].mean(dim=1), label=model_name, c=color_map[model_name])
        axs[1, 1].plot(all_val_accuracies[model_name].mean(dim=1), label=model_name, c=color_map[model_name])
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    time_string = datetime.now().strftime("%m:%d:%Y-%H:%M:%S")
    fig.savefig(f'Code/Figures/loss_accuracy_{time_string}.pdf', bbox_inches='tight')

    plt.close(fig)
