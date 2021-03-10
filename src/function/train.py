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
        optimizer = opt.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        scheduler = opt.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=self.gamma)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        min_loss = 99999
        renew_epoch = 0
        torch.autograd.set_detect_anomaly(True)
        with SummaryWriter(log_dir=Path(self.log_dir)) as writer:
            for epoch in range(self.last_epoch + 1, self.last_epoch + self.epochs + 1):
                running_loss = 0.0
                running_loss_loc = running_loss_obj = running_loss_noobj = running_loss_c = 0
                with tqdm(self.dataloader, total=len(self.dataloader)) as pbar:
                    for images, gts, masks in pbar:
                        # to GPU device
                        images = images.to(device)
                        gts = gts.to(device)
                        masks = masks.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self.model(images)
                        loss_loc, loss_obj, loss_noobj, loss_c = self.model.loss(outputs, gts, masks, self.coefs)
                        loss = loss_loc + loss_obj + loss_noobj + loss_c
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        running_loss_loc += loss_loc.item()
                        running_loss_obj += loss_obj.item()
                        running_loss_noobj += loss_noobj.item()
                        running_loss_c += loss_c.item()

                        # description
                        pbar.set_description(f'[Epoch {epoch}/{self.last_epoch + self.epochs}] loss: {running_loss / (pbar.n + 1)}')

                if running_loss < min_loss:
                    weights_dir = Path(self.weights_dir)
                    for weights_file in weights_dir.glob(f'{self.key}-*.pt'):
                        weights_file.unlink()
                    save_path = weights_dir / f'{self.key}-{epoch:05}.pt'
                    torch.save(self.model.state_dict(), save_path)
                    min_loss = running_loss
                    renew_epoch = epoch

                if epoch - renew_epoch > self.no_change_limit:
                    scheduler.step()
                    renew_epoch = epoch

                writer.add_scalar('loss', running_loss, epoch)
                writer.add_scalar('loss/localization', running_loss_loc, epoch)
                writer.add_scalar('loss/object', running_loss_obj, epoch)
                writer.add_scalar('loss/no object', running_loss_noobj, epoch)
                writer.add_scalar('loss/classification', running_loss_c, epoch)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        print('Finished Training')
