import torch
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path


class Train:
    def __init__(self, model, dataloader, **cfg):
        self.model = model
        self.dataloader = dataloader

        self.__dict__.update(cfg)

    def run(self):
        optimizer = opt.SGD(self.model.get_paramaters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        scheduler = opt.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.steps, gamma=self.gamma)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        min_loss = 99999
        with SummaryWriter(log_dir=Path(self.log_dir)) as writer:
            for epoch in range(self.last_epoch, self.last_epoch + self.epochs):
                self.dataloader.__init__()
                running_loss = 0.0
                with tqdm(self.dataloader, total=len(self.dataloader)) as pbar:
                    for images, gts, masks in pbar:
                        # description
                        pbar.set_description(f'[Epoch {epoch+1}/{self.epochs}] loss: {running_loss}')

                        # to GPU device
                        images = images.to(device)
                        gts = gts.to(device)
                        masks = masks.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self.model(images)
                        loss = self.model.loss(outputs, gts, masks)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    if running_loss < min_loss:
                        save_path = Path(self.assets_dir) / f'{self.key}-{epoch:05}.pt'
                        torch.save(self.model.state_dict(), save_path)
                        min_loss = running_loss

                    scheduler.step()
                    writer.add_scalar('loss', running_loss, epoch)
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        print('Finished Training')
