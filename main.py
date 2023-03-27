from helper import *
from Unet import UNet
from discriminator import Discriminator
from getLoss import get_gen_loss

def main():
    adv_criterion = nn.BCEWithLogitsLoss() # Binary Cross Entroy Loss.
    recon_criterion = nn.L1Loss() # L1 Loss = MAE in Reconstruction
    lambda_recon = 200

    n_epochs = 160
    input_dim = 3
    real_dim = 3
    display_step = 100
    batch_size = 1
    lr = 0.0003
    target_shape = 256
    device = 'cuda'
    save_model = True

    transform = transforms.Compose([
        transforms.ToTensor(),         # jpg to Tensor Converter.
    ])
    dataset = torchvision.datasets.ImageFolder("maps", transform=transform) # Prepares Dataset to be loaded in the form of Pytorch Tensors.

    gen = UNet(input_dim, real_dim).to(device) # Creating instance of generator and moving to device (CUDA is faster).
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr) # Using ADAM to optimize generator.
    disc = Discriminator(input_dim + real_dim).to(device) # Create instance of Discriminator and move to device (CUDA is faster).
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr) # Using ADAM to optimize discriminator.
    gen.apply(weights_init) # Initialize Weights
    disc.apply(weights_init) # Initialize Weights

    mean_generator_loss = 0
    counter = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        for image, _ in tqdm(dataloader):
            image_width = image.shape[3]
            condition = image[:, :, :, :image_width // 2]
            condition = nn.functional.interpolate(condition, size=target_shape)
            real = image[:, :, :, image_width // 2:]
            real = nn.functional.interpolate(real, size=target_shape)
            condition = condition.to(device)
            real = real.to(device)

            disc_opt.zero_grad()
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(), condition) 
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) 
            disc_opt.step() 

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                #a = show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                #b = show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                #c = show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                #d = torch.cat([a,b,c], axis = 1)
                #plt.imshow(d)
                #plt.show()
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                if save_model:
                    if counter > 5:
                        counter = 0
                    counter += 1
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"test{counter}.pth")
            cur_step += 1

if __name__ == '__main__':
    main()