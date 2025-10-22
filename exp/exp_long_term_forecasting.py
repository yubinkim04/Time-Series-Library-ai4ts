from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import wandb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ------------------ Weights & Biases: organized initialization ------------------
        # Build a clean run name & group for organization
        run_name = getattr(self.args, "model_id", None)
        if run_name is None:
            run_name = f"{self.args.data}_{self.args.seq_len}_{self.args.pred_len}"

        group = os.getenv("WANDB_GROUP", f"{self.args.data}_S{self.args.seq_len}_P{self.args.pred_len}")
        tags = [
            f"data:{self.args.data}",
            f"features:{self.args.features}",
            f"seq:{self.args.seq_len}",
            f"pred:{self.args.pred_len}",
            f"el:{self.args.e_layers}",
            f"dl:{self.args.d_layers}",
            f"dm:{self.args.d_model}",
            f"dff:{self.args.d_ff}",
        ]

        # run = wandb.init(
        #     project=os.getenv("WANDB_PROJECT", str(self.args.model)),
        #     name=run_name,
        #     group=group,
        #     job_type="train",
        #     tags=tags,
        #     config=vars(self.args),
        #     reinit=True,  # safe if multiple runs in same process
        #     settings=wandb.Settings(init_timeout=300)
        # )
        run = wandb.init(
            project=f"{os.getenv('WANDB_PROJECT', str(self.args.model))}_P{self.args.pred_len}",
            name=run_name,
            group=group,
            job_type="train",
            tags=tags,
            config=vars(self.args),
            reinit=True,  # safe if multiple runs in same process
            settings=wandb.Settings(init_timeout=300)
        )
        # Log model graph/gradients periodically
        wandb.watch(self.model, log="all", log_freq=200)
        wandb.config.model_architecture = type(self.model).__name__

        # Define metrics so W&B shows minima and aligns by epoch
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("train_loss", summary="min")
            wandb.define_metric("vali_loss", summary="min")
            wandb.define_metric("test_loss", summary="min")
            wandb.define_metric("lr", step_metric="epoch")
        except Exception as _e:
            # Older wandb versions may not support define_metric
            pass
        # -------------------------------------------------------------------------------

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # Log metrics with explicit epoch step and learning rate for clean curves
            try:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "test_loss": test_loss,
                    "lr": model_optim.param_groups[0]["lr"],
                }, step=epoch + 1)
            except Exception as _e:
                pass

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                try:
                    wandb.run.summary["early_stopped_epoch"] = epoch + 1
                except Exception as _e:
                    pass
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Log the best checkpoint as a W&B artifact for easy download/comparison
        try:
            artifact = wandb.Artifact(f"{run_name}-ckpt", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[WARN] Could not log model artifact: {e}")

        # return trained (best) model; run is intentionally left open for test() logging
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Per-lead plots to W&B
        for i in range(self.args.pred_len):
            pred = preds[:, i, :]
            true = trues[:, i, :]
            fig, ax = plt.subplots()
            ax.plot(pred.flatten(), label=f'lead time: {i+1}')
            ax.plot(true.flatten(), label=f'trues')
            ax.legend()
            try:
                wandb.log({"plot": wandb.Image(fig)})
            except Exception as _e:
                pass

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
        else:
            dtw_val = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        try:
            wandb.run.summary["mae"] = mae
            wandb.run.summary["mse"] = mse
            wandb.run.summary["rmse"] = rmse
            wandb.run.summary["mape"] = mape
            wandb.run.summary["mspe"] = mspe
        except Exception as _e:
            pass
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Ensure key metrics & dtw land in the W&B summary and close the run cleanly
        try:
            wandb.run.summary.update({
                "mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe, "dtw": dtw_val
            })
            wandb.finish()
        except Exception as _e:
            pass

        return




# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.metrics import metric
# import torch
# import torch.nn as nn
# from torch import optim
# import os
# import time
# import warnings
# import numpy as np
# from utils.dtw_metric import dtw, accelerated_dtw
# from utils.augmentation import run_augmentation, run_augmentation_single
# import wandb
# import matplotlib.pyplot as plt

# warnings.filterwarnings('ignore')


# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion
 

#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

#                 pred = outputs.detach()
#                 true = batch_y.detach()

#                 loss = criterion(pred, true)

#                 total_loss.append(loss.item())
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         run = wandb.init(project=str(self.args.model), config=vars(self.args))
#         wandb.watch(self.model)
#         wandb.config.model_architecture = str(self.model)


#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                     loss = criterion(outputs, batch_y)
#                     train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             test_loss = self.vali(test_data, test_loader, criterion)

#             wandb.log({
#                 "train_loss": train_loss, 
#                 "vali_loss": vali_loss, 
#                 "test_loss": test_loss, 
#             })

#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#             adjust_learning_rate(model_optim, epoch + 1, self.args)

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         # run.log_model(path=best_model_path, name="best_model")

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, :]
#                 batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = batch_y.shape
#                     if outputs.shape[-1] != batch_y.shape[-1]:
#                         outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
#                     outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

#                 outputs = outputs[:, :, f_dim:]
#                 batch_y = batch_y[:, :, f_dim:]

#                 pred = outputs
#                 true = batch_y

#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     if test_data.scale and self.args.inverse:
#                         shape = input.shape
#                         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#         preds = np.concatenate(preds, axis=0)
#         trues = np.concatenate(trues, axis=0)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)

#         for i in range(self.args.pred_len):
#             pred = preds[:,i,:]
#             true = trues[:,i,:]
#             fig, ax = plt.subplots()
#             ax.plot(pred.flatten(), label=f'lead time: {i+1}')
#             ax.plot(true.flatten(), label=f'trues')
#             ax.legend()
#             wandb.log({"plot": wandb.Image(fig)})
        
#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         # dtw calculation
#         if self.args.use_dtw:
#             dtw_list = []
#             manhattan_distance = lambda x, y: np.abs(x - y)
#             for i in range(preds.shape[0]):
#                 x = preds[i].reshape(-1, 1)
#                 y = trues[i].reshape(-1, 1)
#                 if i % 100 == 0:
#                     print("calculating dtw iter:", i)
#                 d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
#                 dtw_list.append(d)
#             dtw = np.array(dtw_list).mean()
#         else:
#             dtw = 'Not calculated'

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         wandb.run.summary["mae"] = mae
#         wandb.run.summary["mse"] = mse
#         wandb.run.summary["rmse"] = rmse
#         wandb.run.summary["mape"] = mape
#         wandb.run.summary["mspe"] = mspe
#         print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
#         f = open("result_long_term_forecast.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)

#         return
