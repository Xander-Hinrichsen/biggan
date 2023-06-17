import torch
import torch.nn as nn
from models.DeepGanResnets import ResblockDown, ResblockUP
import numpy as np

device = torch.device('cuda')

class Generator(nn.Module):
    def __init__(self, zdim=128, ch=128, classdim=128, numclasses=10, resolution=32):
        super().__init__()
        self.classembeddings = nn.Embedding(num_embeddings=numclasses, embedding_dim=classdim)

        self.biglin = nn.Linear(zdim+classdim, 4*4*8*ch)
        if resolution==32:
            self.biglin = nn.Linear(zdim+classdim, 4*4*8*ch)
            self.start_channels = 8*ch
            self.network = nn.Sequential(
                ResblockUP(8*ch, neuter=True),
                ResblockUP(8*ch, channelreducefactor=2),
                ResblockUP(4*ch, neuter=True),
                ResblockUP(4*ch, channelreducefactor=2),
                ResblockUP(2*ch, neuter=True),
                ResblockUP(2*ch, channelreducefactor=2),
            )
        elif resolution==45:
            self.biglin = nn.Linear(zdim+classdim, 4*4*16*ch)
            self.start_channels = 16*ch
            self.network = nn.Sequential(
                ResblockUP(16*ch, neuter=True),
                ResblockUP(16*ch, channelreducefactor=2),
                ResblockUP(8*ch, neuter=True),
                ResblockUP(8*ch, channelreducefactor=2),
                ResblockUP(4*ch, neuter=True),
                ResblockUP(4*ch, channelreducefactor=2),
                ResblockUP(2*ch, neuter=True),
                ResblockUP(2*ch, channelreducefactor=2, scale_factor=1.40625), #output to 45x45
            )
        elif resolution==128:
            self.biglin = nn.Linear(zdim+classdim, 4*4*16*ch)
            self.start_channels = 16*ch
            self.network = nn.Sequential(
                ResblockUP(16*ch, neuter=True),
                ResblockUP(16*ch, channelreducefactor=1),
                ResblockUP(16*ch, neuter=True),
                ResblockUP(16*ch, channelreducefactor=2),
                ResblockUP(8*ch, neuter=True),
                ResblockUP(8*ch, channelreducefactor=2),
                ResblockUP(4*ch, neuter=True),
                ResblockUP(4*ch, channelreducefactor=2),
                ResblockUP(2*ch, neuter=True),
                ResblockUP(2*ch, channelreducefactor=2),
            )
        elif resolution == 256 or 512:
            self.biglin = nn.Linear(zdim+classdim, 4*4*16*ch)
            self.start_channels = 16*ch
            self.network = nn.Sequential(
                ResblockUP(16*ch, neuter=True),
                ResblockUP(16*ch, channelreducefactor=1),
                ResblockUP(16*ch, neuter=True),
                ResblockUP(16*ch, channelreducefactor=2),
                ResblockUP(8*ch, neuter=True),
                ResblockUP(8*ch, channelreducefactor=1),
                ResblockUP(8*ch, neuter=True),
                ResblockUP(8*ch, channelreducefactor=2),
                ResblockUP(4*ch, neuter=True),
                ResblockUP(4*ch, channelreducefactor=2),
                ResblockUP(2*ch, neuter=True),
                ResblockUP(2*ch, channelreducefactor=2),
            )
        if resolution == 512:
            self.network.append(nn.Sequential(
                ResblockUP(ch, neuter=True),
                ResblockUP(ch, channelreducefactor=1)
            ))
        
        self.network.append(nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, 3, kernel_size=(3,3), padding=1),
            nn.Tanh()
        ))
    def forward(self, xb, labels):
        #embeddings = self.classembeddings(labels)
        #xb = torch.cat((xb, embeddings), dim=1)
        xb = self.biglin(xb).view(-1, self.start_channels, 4, 4)
        return self.network(xb)
    
class Discriminator(nn.Module):
    def __init__(self, resolution=32, ch=128, numclasses=10):
        super().__init__()
        self.numclasses = numclasses
        if resolution == 32 or 45:
            self.network = nn.Sequential(
                nn.Conv2d(3, ch, kernel_size=(3,3), padding=1),
                ResblockDown(ch, channelincreasefactor=2),
                ResblockDown(2*ch, neuter=True),
                ResblockDown(2*ch, channelincreasefactor=2),
                ResblockDown(4*ch, neuter=True),
                ResblockDown(4*ch, channelincreasefactor=2),
                ResblockDown(8*ch, neuter=True),
                nn.ReLU(), 
            )
        elif resolution == 128:
            self.network = nn.Sequential(
                nn.Conv2d(3, ch, kernel_size=(3,3), padding=1),
                ResblockDown(ch, channelincreasefactor=2),
                ResblockDown(2*ch, neuter=True),
                ResblockDown(2*ch, channelincreasefactor=2),
                ResblockDown(4*ch, neuter=True),
                ResblockDown(4*ch, channelincreasefactor=2),
                ResblockDown(8*ch, neuter=True),
                ResblockDown(8*ch, channelincreasefactor=2),
                ResblockDown(16*ch, neuter=True),
                ResblockDown(16*ch, channelincreasefactor=1),
                ResblockDown(16*ch, neuter=True),
                nn.ReLU(), 
            )
        elif resolution == 256:
              self.network = nn.Sequential(
                nn.Conv2d(3, ch, kernel_size=(3,3), padding=1),
                ResblockDown(ch, channelincreasefactor=2),
                ResblockDown(2*ch, neuter=True),
                ResblockDown(2*ch, channelincreasefactor=2),
                ResblockDown(4*ch, neuter=True),
                ResblockDown(4*ch, channelincreasefactor=2),
                ResblockDown(8*ch, neuter=True),
                ResblockDown(8*ch, channelincreasefactor=1),
                ResblockDown(8*ch, neuter=True),
                ResblockDown(8*ch, channelincreasefactor=2),
                ResblockDown(16*ch, neuter=True),
                ResblockDown(16*ch, channelincreasefactor=1),
                ResblockDown(16*ch, neuter=True),
                nn.ReLU(),  
            )
        elif resolution == 512:
              self.network = nn.Sequential(
                nn.Conv2d(3, ch, kernel_size=(3,3), padding=1),
                ResblockDown(ch, channelincreasefactor=1),
                ResblockDown(ch, neuter=True),
                ResblockDown(ch, channelincreasefactor=2),
                ResblockDown(2*ch, neuter=True),
                ResblockDown(2*ch, channelincreasefactor=2),
                ResblockDown(4*ch, neuter=True),
                ResblockDown(4*ch, channelincreasefactor=2),
                ResblockDown(8*ch, neuter=True),
                ResblockDown(8*ch, channelincreasefactor=1),
                ResblockDown(8*ch, neuter=True),
                ResblockDown(8*ch, channelincreasefactor=2),
                ResblockDown(16*ch, neuter=True),
                ResblockDown(16*ch, channelincreasefactor=1),
                ResblockDown(16*ch, neuter=True),
                nn.ReLU(),    
            )
    
        self.class_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear((8*ch) if (resolution==32 or resolution==45) else (16*ch), numclasses)
        )
        if resolution == 45:
            self.classifier = nn.Sequential(
              nn.Flatten(),
              nn.Linear((8*ch*5*5), 1),
              nn.Sigmoid()
            )
        else: 
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear((8*ch*4*4) if resolution==32 else (16*ch*4*4), 1),
                nn.Sigmoid()
            )
        self.sigmoid = nn.Sigmoid()
    def forward(self, xb):
        xb = self.network(xb)
        return self.classifier(xb), self.class_classifier(xb)

class DeepGAN(nn.Module):
    def __init__(self, zdim=128, ch=128, classdim=128, numclasses=10, resolution=32):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.G = Generator(zdim=zdim,ch=ch,classdim=classdim,numclasses=numclasses,resolution=resolution)
        self.D = Discriminator(resolution=resolution, ch=ch, numclasses=numclasses)
        self.numclasses = numclasses
        self.zdim = zdim
        self.cross_entropy= nn.CrossEntropyLoss()
    def forward(self):
        print('use forwards of self.G and self.D')

    def D_trainstep(self, xb, labels, batch_size, optim):
        real = torch.ones(batch_size).to(device)
        fake = torch.zeros(batch_size).to(device)

        z = torch.randn(batch_size, self.zdim*2).to(device)
        fake_labels = torch.randint(low=0, high=self.numclasses, size=(batch_size,)).long().to(device)
        fake_imgs = self.G(z, fake_labels)

        gen_preds, _ = self.D(fake_imgs)
        real_preds, class_preds = self.D(xb)

        discrim_loss = self.bce_loss(real_preds.view(-1), real) + self.bce_loss(gen_preds.view(-1), fake)
        #class_loss = self.cross_entropy(class_preds, labels)

        total_loss = discrim_loss #+ class_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.D.parameters(), 0.1)
        optim.step()
        return total_loss, self.accuracy(class_preds, labels)

    def G_trainstep(self, batch_size, optim, num_samples=5):
        real = torch.ones(batch_size).to(device)
        z = torch.randn(batch_size, self.zdim*2).to(device)
        fake_labels = torch.randint(low=0, high=self.numclasses, size=(batch_size,)).long().to(device)
        fake_imgs = self.G(z, fake_labels)

        gen_preds, class_preds = self.D(fake_imgs)

        gen_loss = self.bce_loss(gen_preds.view(-1), real)
        #class_loss = self.cross_entropy(class_preds, fake_labels)

        total_loss = gen_loss #+ class_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.G.parameters(), 0.1)
        optim.step()

        return total_loss, self.accuracy(class_preds, fake_labels), fake_imgs[0:num_samples,:,:]
    
    def D_valstep(self,xb, labels, batch_size):
        real = torch.ones(batch_size).to(device)
        fake = torch.zeros(batch_size).to(device)
        z = torch.randn(batch_size, self.zdim*2).to(device)
        fake_labels = torch.randint(low=0, high=self.numclasses, size=(batch_size,)).long().to(device)
        fake_imgs = self.G(z, fake_labels)

        gen_preds, _ = self.D(fake_imgs)
        real_preds, class_preds = self.D(xb)

        discrim_loss = self.bce_loss(real_preds.view(-1), real) + self.bce_loss(gen_preds.view(-1), fake)
        #class_loss = self.cross_entropy(class_preds, labels)

        total_loss = discrim_loss #+ class_loss

        return total_loss, self.accuracy(class_preds, labels)
    
    def accuracy(self, preds, truth):
        with torch.no_grad():
          preds = torch.argmax(preds, dim=1)
          return (torch.sum(preds==truth) / preds.shape[0]).item()