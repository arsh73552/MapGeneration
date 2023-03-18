import torch

def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    fake_img = gen(condition)
    disc_pred = disc(fake_img, condition)
    adv_loss = adv_criterion(disc_pred, torch.ones_like(disc_pred))
    rec_loss = recon_criterion(real, fake_img)
    gen_loss = adv_loss + lambda_recon * rec_loss
    return gen_loss