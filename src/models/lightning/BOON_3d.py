import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from ..multi_step.BOON_3d import BOON_FNO3d
from ..base.FNO3d import FNO3d

class BOON3D(pl.LightningModule):
    def __init__(
        self,
        net_name="test1",
        in_channels=1,
        out_channels=1,
        modes1=8,
        modes2=8,
        modes3=8,
        width=20,
        lr=1e-3,
        num_layers=4
    ):
        super(BOON3D, self).__init__()
        
        self.save_hyperparameters()
        
        self.net_name = net_name
        self.lr = lr
        self.fno = FNO3d(
            #in_channels=in_channels,
            #out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            lb=0,
            ub=256,
            #num_layers=num_layers
        )
        
        self.model = BOON_FNO3d(
            width = width,
            base_no = self.fno,
            lb = 0,
            ub = 256,
            bdy_type = 'dirichlet'
        )
        

    def forward(self, x, bc_left=None, bc_right=None, bc_top=None, bc_bottom=None):
        return self.model(x, bdy_left=bc_left, bdy_right=bc_right, bdy_top=bc_top, bdy_down=bc_bottom)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        bs, nx, ny, nz, _ = x.shape
        #bdy_left  = x[:, 0, :, :, :].reshape(bs, 1, ny, nz)
        #bdy_right = x[:,-1, :, :, :].reshape(bs, 1, ny, nz)
        #bdy_top = x[:, :, 0, :, :].reshape(bs, 1, nx, nz)
        #bdy_down = x[:, :, -1, :, :].reshape(bs, 1, nx, nz)

        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        bdy_top = y[:, :, :, 0, :].reshape(bs, 1, nx, nz)
        bdy_down = y[:, :, :, -1, :].reshape(bs, 1, nx, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right}, bc_top={'val': bdy_top}, bc_bottom={'val': bdy_down})
        # TODO: Mask the grains
        #mask = (x != 0).to(yhat.dtype)
        #yhat = yhat * mask
        
        loss = 0
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)), reduction='none')
        mask = (x != 0).to(yhat.dtype).reshape((1, -1))
        weighted_loss = loss * mask
        active = mask.sum()
        loss = weighted_loss.sum() / torch.clamp(active, min=1.0)
        

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        bs, nx, ny, nz, _ = x.shape
        #bdy_left  = x[:, 0, :, :, :].reshape(bs, 1, ny, nz)
        #bdy_right = x[:,-1, :, :, :].reshape(bs, 1, ny, nz)
        #bdy_top = x[:, :, 0, :, :].reshape(bs, 1, nx, nz)
        #bdy_down = x[:, :, -1, :, :].reshape(bs, 1, nx, nz)

        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        bdy_top = y[:, :, :, 0, :].reshape(bs, 1, nx, nz)
        bdy_down = y[:, :, :, -1, :].reshape(bs, 1, nx, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right}, bc_top={'val': bdy_top}, bc_bottom={'val': bdy_down})
        #mask = (x != 0).to(yhat.dtype)
        #yhat = yhat * mask

        loss = 0
        #loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)), reduction='none')
        mask = (x != 0).to(yhat.dtype).reshape((1, -1))
        weighted_loss = loss * mask
        active = mask.sum()
        loss = weighted_loss.sum() / torch.clamp(active, min=1.0)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            rank_zero_only=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        bs, nx, ny, nz, _ = x.shape
        bdy_left  = x[:, 0, :, :, :].reshape(bs, 1, ny, nz)
        bdy_right = x[:,-1, :, :, :].reshape(bs, 1, ny, nz)
        bdy_top = x[:, :, 0, :, :].reshape(bs, 1, nx, nz)
        bdy_down = x[:, :, -1, :, :].reshape(bs, 1, nx, nz)


        #bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        #bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        #bdy_top = y[:, :, :, 0, :].reshape(bs, 1, nx, nz)
        #bdy_down = y[:, :, :, -1, :].reshape(bs, 1, nx, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right}, bc_top={'val': bdy_top}, bc_bottom={'val': bdy_down})
        #mask = (x != 0).to(yhat.dtype)
        #yhat = yhat * mask
        

        loss = 0
        #loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)), reduction='none')
        mask = (x != 0).to(yhat.dtype).reshape((1, -1))
        weighted_loss = loss * mask
        active = mask.sum()
        loss = weighted_loss.sum() / torch.clamp(active, min=1.0)
        #variance = y.detach().var(correction=0)
        #loss = loss / (variance.clamp_min(1e-6) + 1e-8)
        
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            rank_zero_only=True,
        )

        test_metrics = {"loss": loss,}

        return test_metrics

    def predict_step(self, batch, batch_idx):
        x, y = batch
        bs, nx, ny, nz, _ = x.shape
        #bdy_left  = x[:, 0, :, :, :].reshape(bs, 1, ny, nz)
        #bdy_right = x[:,-1, :, :, :].reshape(bs, 1, ny, nz)
        #bdy_top = x[:, :, 0, :, :].reshape(bs, 1, nx, nz)
        #bdy_down = x[:, :, -1, :, :].reshape(bs, 1, nx, nz)


        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        bdy_top = y[:, :, :, 0, :].reshape(bs, 1, nx, nz)
        bdy_down = y[:, :, :, -1, :].reshape(bs, 1, nx, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right}, bc_top={'val': bdy_top}, bc_bottom={'val': bdy_down})
        #mask = (x != 0).to(yhat.dtype)
        #yhat = yhat * mask
        
        loss = 0
        #loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)), reduction='none')
        mask = (x != 0).to(yhat.dtype).reshape((1, -1))
        weighted_loss = loss * mask
        active = mask.sum()
        loss = weighted_loss.sum() / torch.clamp(active, min=1.0)
                
        predictions = {
            "x": x,
            "y": y,
            "yhat": yhat,
        }

        return predictions

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, min_lr=0.5 * self.lr),
                     'monitor': 'train_loss'}
        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr / 10)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        
