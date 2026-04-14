import numpy as np
import torch
import torch.nn as nn

import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class GradientInversionAttack():

    def __init__(self, method="iDLG", iters=2000, lr=1.0, device="cpu", save_dir="attack_results"):
        self.method = method
        self.iters = iters
        self.lr = lr
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

    def reconstruct(
        self,
        model,
        gt_shape,
        num_classes,
        gt_data=None,
        gt_label=None,
    ):
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        # =========================
        # STEP 1: GET REAL GRADIENT
        # =========================
        gt_data = gt_data.to(self.device)
        gt_label = gt_label.to(self.device)

        out = model(gt_data)
        loss = criterion(out, gt_label)

        gradients = torch.autograd.grad(loss, model.parameters())
        gradients = [g.detach().clone() for g in gradients]
        
        # =========================
        # INIT DUMMY
        # =========================
        dummy_data = torch.randn(gt_shape, device=self.device, requires_grad=True)

        if self.method == "DLG":
            dummy_label = torch.randn((gt_shape[0], num_classes), device=self.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.lr)
        else:
            last_weight_grad = gradients[-2]
            label_pred = torch.argmin(
                torch.sum(last_weight_grad, dim=-1), dim=-1
            ).detach().view(1)

            optimizer = torch.optim.LBFGS([dummy_data], lr=self.lr)

        losses = []
        mses = []
        history = []
        history_iters = []

        # =========================
        # OPTIMIZATION
        # =========================
        for it in range(self.iters):

            def closure():
                optimizer.zero_grad()
                pred = model(dummy_data)

                if self.method == "DLG":
                    loss = - torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) *
                                torch.log(torch.softmax(pred, -1)), dim=-1)
                    )
                else:
                    loss = criterion(pred, label_pred.view(-1))

                dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_grads, gradients):
                    grad_diff += ((gx - gy) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            loss = optimizer.step(closure)
            current_loss = loss.item()
            
            losses.append(current_loss)

            if gt_data is not None:
                mse = torch.mean((dummy_data - gt_data) ** 2).item()
                mses.append(mse)

            # =========================
            # SAVE HISTORY
            # =========================
            if it % max(1, self.iters // 10) == 0:
                print(f"Iter {it} | Loss: {current_loss:.6f}" +
                    (f" | MSE: {mses[-1]:.6f}" if len(mses) > 0 else ""))

                history.append(dummy_data.clone().detach().cpu())
                history_iters.append(it)

        # =========================
        # SAVE FINAL IMAGE
        # =========================
        with torch.no_grad():
            img = dummy_data.clone().detach().cpu()
            vutils.save_image(img, os.path.join(self.save_dir, "attack.png"), normalize=True)

        # =========================
        # SAVE ITER IMAGES
        # =========================
        for t, img in zip(history_iters, history):
            vutils.save_image(
                img,
                os.path.join(self.save_dir, f"iter_{t}.png"),
                normalize=True
            )

        # =========================
        # PROGRESSION PLOT (GIỐNG PAPER)
        # =========================

        num_steps = len(history)

        for i in range(gt_shape[0]):
            plt.figure(figsize=(12, 3))

            # GT
            if gt_data is not None:
                plt.subplot(1, num_steps + 1, 1)
                plt.imshow(gt_data[i].cpu().squeeze(), cmap="gray")
                plt.title("GT")
                plt.axis("off")

            # History
            for t in range(num_steps):
                plt.subplot(1, num_steps + 1, t + 2)
                plt.imshow(history[t][i].squeeze(), cmap="gray")
                plt.title(f"{history_iters[t]}")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"progress_sample_{i}.png"))
            plt.close()

        # =========================
        # LOSS PLOT
        # =========================
        plt.figure()
        plt.plot(losses, label="Gradient Loss")

        if len(mses) > 0:
            plt.plot(mses, label="MSE")

        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Attack")

        plt.savefig(os.path.join(self.save_dir, "attack_plot.png"))
        plt.close()

        return dummy_data.detach()
    
    def reconstruct_multiple(
        self,
        model,
        data_list,   # <-- list [(x, y), (x, y), ...]
        num_classes
    ):
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        gt_list = []
        recon_list = []

        for i, (gt_data, gt_label) in enumerate(data_list):
            print(f"\n=== Attack sample {i} ===")

            gt_data = gt_data.to(self.device)
            gt_label = gt_label.to(self.device)

            # =========================
            # REAL GRADIENT
            # =========================
            out = model(gt_data)
            loss = criterion(out, gt_label)

            gradients = torch.autograd.grad(loss, model.parameters())
            gradients = [g.detach().clone() for g in gradients]

            # =========================
            # INIT DUMMY
            # =========================
            dummy_data = torch.randn_like(gt_data, requires_grad=True)

            if self.method == "DLG":
                dummy_label = torch.randn((gt_data.shape[0], num_classes), device=self.device, requires_grad=True)
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.lr)
            else:
                last_weight_grad = gradients[-2]
                label_pred = torch.argmin(
                    torch.sum(last_weight_grad, dim=-1), dim=-1
                ).detach().view(1)

                optimizer = torch.optim.LBFGS([dummy_data], lr=self.lr)

            # =========================
            # OPTIMIZATION
            # =========================
            for _ in range(self.iters):

                def closure():
                    optimizer.zero_grad()
                    pred = model(dummy_data)

                    if self.method == "DLG":
                        loss = - torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) *
                                    torch.log(torch.softmax(pred, -1)), dim=-1)
                        )
                    else:
                        loss = criterion(pred, label_pred.view(-1))

                    dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_grads, gradients):
                        grad_diff += ((gx - gy) ** 2).sum()

                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)

            gt_list.append(gt_data.detach().cpu())
            recon_list.append(dummy_data.detach().cpu())

        # =========================
        # SAVE PAIRS (GT -> RECON)
        # =========================
        for i in range(len(gt_list)):
            gt = gt_list[i][0].squeeze()
            recon = recon_list[i][0].squeeze()

            plt.figure(figsize=(4, 2))

            # GT
            plt.subplot(1, 2, 1)
            plt.imshow(gt, cmap="gray")
            plt.title("GT")
            plt.axis("off")

            # RECON
            plt.subplot(1, 2, 2)
            plt.imshow(recon, cmap="gray")
            plt.title("Recon")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"pair_{i}.png"))
            plt.close()

        # =========================
        # PLOT GRID
        # =========================
        n = len(gt_list)
        ncols = 10
        nrows = (n + ncols - 1) // ncols
        
        plt.figure(figsize=(2*ncols, 2*nrows*2))

        # ===== GT =====
        for i in range(n):
            plt.subplot(2*nrows, ncols, i + 1)
            plt.imshow(gt_list[i][0].squeeze(), cmap="gray")
            plt.title(f"GT {i}")
            plt.axis("off")

        # ===== RECON =====
        for i in range(n):
            plt.subplot(2*nrows, ncols, nrows*ncols + i + 1)
            plt.imshow(recon_list[i][0].squeeze(), cmap="gray")
            plt.title(f"Recon {i}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "multi_attack.png"))
        plt.close()

        return recon_list