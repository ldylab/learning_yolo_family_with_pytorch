import torch
import torch.optim
import torch.utils.data
import time
import warnings

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.yolo_tools import gt_creator
import torch.backends.cudnn as cudnn
from utils.vocapi_evaluator import VOCAPIEvaluator
from data.voc_dataset_transform import BaseTransform
from data.voc_dataset import VOC_CLASSES, VOC_ROOT

# avoiding the warning output
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        # generate the train and val dataset
        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        # generate the dataset to calculate the amp
        val_size = 416
        device = torch.device("cuda")
        self.evaluator = VOCAPIEvaluator(data_root=VOC_ROOT, img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )



        self.model = select_model(args)
        # self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        if args.load_model_path != '':
            print("=> using pre-trained weights for DPSNet")
            self.model.load_state_dict(torch.load(args.load_model_path), strict=args.load_not_strict)


        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)

    def train(self):
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            self.val_amp_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)

    # the train code need to be rewrite
    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        ## starting ##
        train_size = 416
        cudnn.benchmark = True
        device = torch.device("cuda")

        t0 = time.time()

        for iter_i, (images, targets) in enumerate(self.train_loader):

            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=train_size,
                                       stride=self.model.stride,
                                       label_lists=targets
                                       )

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # 前向推理和计算损失
            conf_loss, cls_loss, bbox_loss, total_loss = self.model(images, target=targets)

            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # record the loss to scalar
            self.logger.record_scalar("train_conf_loss", conf_loss)
            self.logger.record_scalar("train_cls_loss", cls_loss)
            self.logger.record_scalar("train_bbox_loss", bbox_loss)
            self.logger.record_scalar("train_total_loss", total_loss)

            # monitor training progress
            if iter_i % self.args.print_freq == 0:

                t1 = time.time()
                print('[Epoch %d][Iter %d]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch, iter_i,
                         conf_loss.item(),
                         cls_loss.item(),
                         bbox_loss.item(),
                         total_loss.item(),
                         train_size,
                         t1 - t0),
                      flush=True)

                t0 = time.time()

    def val_amp_per_epoch(self, epoch):
        val_size = 416

        self.model.trainable = False
        self.model.set_grid(val_size)
        self.model.eval()

        # evaluate
        self.evaluator.evaluate(self.model)

        # convert to training mode.
        self.model.trainable = True
        self.model.set_grid(val_size)
        self.model.train()

    # in the validation we can only see the loss, but not the accuracy!
    def val_per_epoch(self, epoch):
        self.model.eval()

        ## starting ##
        train_size = 416
        cudnn.benchmark = True
        device = torch.device("cuda")

        with torch.no_grad():
            t0 = time.time()

            for iter_i, (images, targets) in enumerate(self.val_loader):

                # 制作ceshi标签
                targets = [label.tolist() for label in targets]
                targets = gt_creator(input_size=train_size,
                                     stride=self.model.stride,
                                     label_lists=targets
                                     )

                # to device
                images = images.to(device)
                targets = targets.to(device)

                # 前向推理和计算损失
                conf_loss, cls_loss, bbox_loss, total_loss = self.model(images, target=targets)

                # record the loss to scalar
                self.logger.record_scalar("train_conf_loss", conf_loss)
                self.logger.record_scalar("train_cls_loss", cls_loss)
                self.logger.record_scalar("train_bbox_loss", bbox_loss)
                self.logger.record_scalar("train_total_loss", total_loss)

                # monitor training progress
                if iter_i % self.args.print_freq == 0:
                    t1 = time.time()
                    print('Evaluation [Epoch %d][Iter %d]'
                          '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                          % (epoch, iter_i,
                             conf_loss.item(),
                             cls_loss.item(),
                             bbox_loss.item(),
                             total_loss.item(),
                             train_size,
                             t1 - t0),
                          flush=True)

                    t0 = time.time()


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
