import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from BOON_3d import BOON_FNO3d
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
        super(BOON_FNO3d, self).__init__()
        
        self.save_hyperparameters()
        
        self.net_name = net_name
        self.lr = lr
        self.fno = FNO3d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            num_layers=num_layers
        )
        
        self.model = BOON_FNO3d(
            width = width,
            base_no = self.fno,
            lb = 1,
            ub = 2,
            bdy_type = 'dirichlet'
        )
        

    def forward(self, x, bc_left=None, bc_right=None, bc_top=None, bc_bottom=None):
        return self.model(x, bdy_left=bc_left, bdy_right=bc_right, bdy_top=bc_top, bdy_down=bc_bottom)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        bs, nx, ny, nz, _ = x.shape
        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right})
        
        # TODO: Mask the grains
        
        loss = 0
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        bs, nx, ny, nz, _ = x.shape
        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right})

        loss = 0
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))

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
        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right})


        loss = 0
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))
        #variance = y.detach().var(correction=0)
        #loss = loss / (variance.clamp_min(1e-6) + 1e-8)
        eik_loss = 0.0
        if self.beta_2 != 0:
            self.b2 = self.linear_ramp(self.current_epoch, t0=300, ramp=30, w=self.beta_2)
            eik_loss = self.b2 * self.eikonal_loss(yhat)
        self.log(
            "test_mse_loss", loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True
        )
        self.log(
            "test_eik_loss", eik_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True
        )
        loss = loss + eik_loss
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
        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        yhat = self(x, bc_left={'val': bdy_left}, bc_right={'val': bdy_right})
        yhat = self(x)

        loss = 0
        loss = F.mse_loss(y.view((1, -1)), yhat.view((1, -1)))
        #variance = y.detach().var(correction=0)
        #loss = loss / (variance.clamp_min(1e-6) + 1e-8)
        eik_loss = 0.0
        if self.beta_2 != 0:
            self.b2 = self.linear_ramp(self.current_epoch, t0=300, ramp=30, w=self.beta_2)
            eik_loss = self.b2 * self.eikonal_loss(yhat)

        loss = loss + eik_loss
        
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

        
