import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CTMRIDataset(Dataset):
    def __init__(self, ct_dir, mri_dir, transform=None):
        self.ct_dir = ct_dir
        self.mri_dir = mri_dir
        self.ct_images = os.listdir(ct_dir)
        self.mri_images = os.listdir(mri_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ct_images)

    def __getitem__(self, idx):
        ct_path = os.path.join(self.ct_dir, self.ct_images[idx])
        mri_path = os.path.join(self.mri_dir, self.mri_images[idx])

        ct_image = Image.open(ct_path).convert("RGB")
        mri_image = Image.open(mri_path).convert("RGB")

        if self.transform:
            ct_image = self.transform(ct_image)
            mri_image = self.transform(mri_image)

        return {"ct": ct_image, "mri": mri_image}

# Self-Attention Layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim, filters):
        super(SelfAttention, self).__init__()
        self.filters = filters
        
        # 定義卷積層
        self.query_conv = nn.Conv2d(in_dim, self.filters, kernel_size=1, padding=0)
        self.key_conv = nn.Conv2d(in_dim, self.filters, kernel_size=1, padding=0)
        self.value_conv = nn.Conv2d(in_dim, self.filters, kernel_size=1, padding=0)
        self.out_conv = nn.Conv2d(self.filters, in_dim, kernel_size=1, padding=0)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()

        # 計算 query, key, value
        query = self.query_conv(inputs)  # [batch_size, filters, height, width]
        key = self.key_conv(inputs)      # [batch_size, filters, height, width]
        value = self.value_conv(inputs)  # [batch_size, filters, height, width]
        
        # Reshape for attention calculation (flatten height and width)
        query_reshape = query.view(batch_size, self.filters, -1).permute(0, 2, 1)  # [batch_size, height*width, filters]
        key_reshape = key.view(batch_size, self.filters, -1)  # [batch_size, filters, height*width]
        value_reshape = value.view(batch_size, self.filters, -1)  # [batch_size, filters, height*width]

        # 計算注意力權重
        attention_weights = torch.bmm(query_reshape, key_reshape)  # [batch_size, height*width, height*width]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 計算注意力輸出
        attention_output = torch.bmm(attention_weights, value_reshape.permute(0, 2, 1))  # [batch_size, height*width, filters]
        attention_output = attention_output.permute(0, 2, 1).view(batch_size, self.filters, height, width)  # [batch_size, filters, height, width]

        # 輸出卷積
        output = self.out_conv(attention_output)
        return output


# Generator with VAE and Self-Attention
import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim=128):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)
        self.enc7 = self.conv_block(512, 512)

        # Self-Attention
        self.attention = SelfAttention(512 , filters = 512)

        # Decoder
        self.dec6 = self.upconv_block(512, 512)
        self.dec5 = self.upconv_block(1024, 512)
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(1024, 256)
        self.dec2 = self.upconv_block(512, 128)
        self.dec1 = self.upconv_block(256, 64)
        self.final = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)

        # Self-Attention
        attention = self.attention(enc7)

        # Decoder with skip connections
        dec6 = self.dec6(attention)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec5 = self.dec5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec4 = self.dec4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        output = self.final(dec1)

        return torch.tanh(output)



# Discriminator with Self-Attention
class UNetDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(UNetDiscriminator, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)
        self.enc7 = self.conv_block(512, 512)

        # Self-Attention
        self.attention = SelfAttention(512 , filters = 512)

        # Decoder
        self.dec6 = self.upconv_block(512, 512)
        self.dec5 = self.upconv_block(1024, 512)
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(1024, 256)
        self.dec2 = self.upconv_block(512, 128)
        self.dec1 = self.upconv_block(256, 64)
        self.final = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)

        # Self-Attention
        attention = self.attention(enc7)

        # Decoder with skip connections
        dec6 = self.dec6(attention)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec5 = self.dec5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec4 = self.dec4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        output = self.final(dec1)

        return torch.sigmoid(output)

if __name__ == "__main__":
    # Hyperparameters
    input_channels = 3  # CT (or CT + Feature Map if expanded)
    output_channels = 3  # MRI
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.0002

    # Paths
    train_ct_dir = "ct-to-mri-cgan/Dataset/images/trainA"
    train_mri_dir = "ct-to-mri-cgan/Dataset/images/trainB"
    test_ct_dir = "ct-to-mri-cgan/Dataset/images/testA"
    test_mri_dir = "ct-to-mri-cgan/Dataset/images/testB"

    # Transformations
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


    # Datasets and Dataloaders
    train_dataset = CTMRIDataset(train_ct_dir, train_mri_dir, transform=transform)
    test_dataset = CTMRIDataset(test_ct_dir, test_mri_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    generator = UNetGenerator(input_channels, output_channels).to(device)
    discriminator = UNetDiscriminator(input_channels).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            ct_images = data['ct'].to(device)
            mri_images = data['mri'].to(device)

            # Update Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(mri_images)
            fake_images = generator(ct_images)
            fake_output = discriminator(fake_images.detach())
            d_loss_real = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
            d_loss_fake = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # 梯度剪裁
            optimizer_D.step()

            # Update Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # 梯度剪裁
            optimizer_G.step()

            # Save losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # Save results
            if i % 100 == 0:
                vutils.save_image(fake_images, f"results/generated_mri_{epoch}_{i}.png", normalize=True)
                vutils.save_image(mri_images, f"results/real_mri_{epoch}_{i}.png", normalize=True)

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    print("Training and testing complete. Results saved to 'results/' directory.")

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_plot.png")
    plt.show()
