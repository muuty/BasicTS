import torch


def decomposed_mae(prediction, target, mu_y_hat, sigma_y_hat,
                   mu_y_target, sigma_y_target, null_val=0.0):
    """Loss for decomposed prediction: main MAE + auxiliary scale losses.

    Main loss is in raw space (after inverse Z-score by postprocessing).
    Auxiliary losses are in Z-score space (not inverse-transformed).
    Lambda is set to scale Z-score-space losses (~1.0 magnitude) to be
    roughly 10% of the main loss (~10-12 magnitude).
    """
    # Main prediction loss (raw space)
    main_loss = torch.mean(torch.abs(prediction - target))

    # Auxiliary scale losses (Z-score space)
    mu_loss = torch.mean(torch.abs(mu_y_hat - mu_y_target))
    sigma_loss = torch.mean(torch.abs(sigma_y_hat - sigma_y_target))

    return main_loss + 1.0 * mu_loss + 1.0 * sigma_loss
