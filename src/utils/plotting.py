import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb


def plot_eigenvalues(eig_val, allo_eig_val, allo_manual_eig_val, args):
    """
    Plot the eigenvalues of the ground truth, estimated betas, and manually estimated betas.
    """
    # GT
    plt.figure()
    plt.bar(range(len(eig_val)), eig_val)
    plt.title('GT Eigenvalues')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.savefig('tmp/tmp_fig/GT.png')
    if args.wb.on:
        wandb.log({"figs/GT": wandb.Image(plt)})

    # Allo Betas
    plt.figure()
    plt.bar(range(len(eig_val)), allo_eig_val)
    plt.title('Allo Betas Eigenvalues')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.savefig('tmp/tmp_fig/AlloBetas.png')
    if args.wb.on:
        wandb.log({"figs/AlloBetas": wandb.Image(plt)})

    # Allo Manual Betas
    plt.figure()
    plt.bar(range(len(eig_val)), allo_manual_eig_val)
    plt.title('Allo Manual Eigenvalues')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.savefig('tmp/tmp_fig/AlloManual.png')
    if args.wb.on:
        wandb.log({"figs/AlloManual": wandb.Image(plt)})


def plot_eigenvalue_differences(eig_val, allo_eig_val, allo_manual_eig_val, args):
    """
    Plot the differences between the eigenvalues of the ground truth, estimated betas, and manually estimated betas.
    The metric used for comparison is the absolute difference between the eigenvalues.
    (Note that the eigenvalues are sorted in ascending order.)
    """

    plt.figure()
    plt.bar(range(len(eig_val)), jnp.abs(eig_val - allo_eig_val))
    plt.title('|GT - Allo Betas| Eigenvalue Difference')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Absolute Difference')
    plt.savefig('tmp/tmp_fig/GTvsALLOBetas.png')
    if args.wb.on:
        wandb.log({"figs/diff/GTvsALLOBetas": wandb.Image(plt)})

    plt.figure()
    plt.bar(range(len(eig_val)), jnp.abs(eig_val - allo_manual_eig_val))
    plt.title('|GT - Manual Allo| Eigenvalue Difference')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Absolute Difference')
    plt.savefig('tmp/tmp_fig/GTvsManualALLO.png')
    if args.wb.on:
        wandb.log({"figs/diff/GTvsManualALLO": wandb.Image(plt)})

    # overlay the eigenvalues and the estimated eigenvalues GT vs Allo Betas
    plt.figure()
    width = 0.8

    indices = jnp.arange(len(eig_val))

    plt.bar(indices, allo_eig_val, width=width, color='r', alpha=0.5, label='Allo Betas')
    plt.bar(indices, eig_val, width=width, color='g', label='GT')
    # plt.yscale('log')
    diff = jnp.abs(eig_val - allo_eig_val)
    for i in indices:
        y_pos = max(allo_eig_val[i], eig_val[i])
        plt.text(i, y_pos, f'{diff[i]:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(indices, ['\u03BB{}'.format(i) for i in range(len(eig_val))])

    plt.title('Eigenvalues: GT vs Allo Betas')
    plt.legend(framealpha=0,
               prop={'size': 8})
    plt.savefig('tmp/tmp_fig/barplot_overlayed_betas.png')
    if args.wb.on:
        wandb.log({"figs/diff/GTvsALLOBetas_overlayed": wandb.Image(plt)})

    # overlay plot for GT vs Manual Allo
    plt.figure()
    width = 0.8

    indices = jnp.arange(len(eig_val))

    plt.bar(indices, allo_manual_eig_val, width=width, color='r', alpha=0.5, label='Allo Manual')
    plt.bar(indices, eig_val, width=width, color='g', label='GT')
    # plt.yscale('log')
    diff = jnp.abs(eig_val - allo_manual_eig_val)
    for i in indices:
        y_pos = max(allo_manual_eig_val[i], eig_val[i])
        plt.text(i, y_pos, f'{diff[i]:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(indices, ['\u03BB{}'.format(i) for i in range(len(eig_val))])

    plt.title('Eigenvalues: GT vs Allo Manual')
    plt.legend(framealpha=0,
               prop={'size': 8})
    plt.savefig('tmp/tmp_fig/barplot_overlayed_manual.png')
    if args.wb.on:
        wandb.log({"figs/diff/GTvsManualALLO_overlayed": wandb.Image(plt)})