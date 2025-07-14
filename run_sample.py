import argparse
import os
import numpy as np
import os.path as osp

from misc import pyutils

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--CAM_root", default='dataset/BCD', type=str,
                        help="Dataset floder. Please enter the folder names for T1,T2 images in the IMG_FOLDER_NAME_A "
                             "and IMG_FOLDER_NAME_B sections of the dataloader")
    parser.add_argument("--SAM_A", default="./dataset/BCD/train/A1/", type=str)
    parser.add_argument("--SAM_B", default="./dataset/BCD/train/B1/", type=str,
                        help="Remove the unchanged pixel pairs in the predicted mask, and only use SAM for the "
                             "changed pixel pairs")
    parser.add_argument("--SAM_weight", default="./pth/sam_vit_h_4b8939.pth", type=str,
                        help="sam_vit_h_4b8939.pth")

    # Dataset
    parser.add_argument("--train_list", default="dataset/BCD/train5.txt", type=str)
    parser.add_argument("--val_list", default="dataset/BCD/val5.txt", type=str)
    parser.add_argument("--infer_list", default="dataset/BCD/amn_train_change.txt", type=str)
    parser.add_argument("--amn_list", default="dataset/BCD/amn_train.txt", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=224, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.05, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.25, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    # ReCAM
    parser.add_argument("--recam_num_epoches", default=10, type=int)
    parser.add_argument("--recam_learning_rate", default=0.0005, type=float)
    parser.add_argument("--recam_loss_weight", default=0.85, type=float)
    # To a certain extent, you can control the balance of precision and recall of pseudo label generation,
    # the larger the value, the larger the recall, but the corresponding precision will be reduced.

    # AMN
    parser.add_argument("--amn_network", default="net.resnet50_amn", type=str)
    parser.add_argument("--amn_crop_size", default=256, type=int)
    parser.add_argument("--amn_batch_size", default=16, type=int)
    parser.add_argument("--amn_num_epoches", default=5, type=int)
    parser.add_argument("--eps", default=0.1, type=float)

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.35, type=float)
    parser.add_argument("--conf_bg_thres", default=0.1, type=float)

    # Output Path
    parser.add_argument("--work_space", default="result", type=str)  # set your path
    parser.add_argument("--log_name", default="sample_train", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam", type=str)
    parser.add_argument("--amn_weights_name", default="res50_amn", type=str)
    parser.add_argument("--cam_out_dir", default="cam", type=str)
    parser.add_argument("--recam_out_dir", default="recam", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--amn_cam_out_dir", default="amn_cam", type=str)
    parser.add_argument("--amn_ir_label_out_dir", default="amn_ir_label", type=str)
    parser.add_argument("--recam_weight_dir", default="recam_weight", type=str)
    parser.add_argument("--mask", default="mask", type=str)
    parser.add_argument("--SAMlabel", default="SAMlabel", type=str)

    # Step
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--train_recam_pass", type=str2bool, default=False)
    parser.add_argument("--make_recam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_amn_cam_pass", type=str2bool, default=False)
    parser.add_argument("--train_amn_pass", type=str2bool, default=False)
    parser.add_argument("--make_amn_cam_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=False)
    parser.add_argument("--amn_cam_to_ir_label_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_mask_pass", type=str2bool, default=False)
    parser.add_argument("--eval_mask_pass", type=str2bool, default=False)
    parser.add_argument("--SAM_label_pass", type=str2bool, default=False)
    parser.add_argument("--eval_sam_pass", type=str2bool, default=False)

    args = parser.parse_args()
    args.log_name = osp.join(args.work_space, args.log_name)
    args.cam_weights_name = osp.join(args.work_space, args.cam_weights_name)
    args.amn_weights_name = osp.join(args.work_space, args.amn_weights_name)
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir)
    args.recam_out_dir = osp.join(args.work_space, args.recam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space, args.ir_label_out_dir)
    args.amn_cam_out_dir = osp.join(args.work_space, args.amn_cam_out_dir)
    args.amn_ir_label_out_dir = osp.join(args.work_space, args.amn_ir_label_out_dir)
    args.recam_weight_dir = osp.join(args.work_space, args.recam_weight_dir)
    args.mask = osp.join(args.work_space, args.mask)
    args.SAMlabel = osp.join(args.work_space, args.SAMlabel)

    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    # os.makedirs(args.recam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.amn_cam_out_dir, exist_ok=True)
    os.makedirs(args.amn_ir_label_out_dir, exist_ok=True)
    # os.makedirs(args.recam_weight_dir, exist_ok=True)
    os.makedirs(args.mask, exist_ok=True)
    os.makedirs(args.SAMlabel, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    # print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.train_recam_pass is True:
        import step.train_recam

        timer = pyutils.Timer('step.train_recam:')
        step.train_recam.run(args)

    if args.make_recam_pass is True:
        import step.make_recam

        timer = pyutils.Timer('step.make_recam:')
        step.make_recam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        final_miou = []
        for i in range(30, 35):
            t = i / 100.0
            args.cam_eval_thres = t
            miou = step.eval_cam.run(args)
            final_miou.append(miou)
        # print(args.cam_out_dir)
        # print(final_miou)
        print(np.max(np.array(final_miou)))

    if args.train_amn_pass is True:
        import step.train_amn

        timer = pyutils.Timer('step.train_amn:')
        step.train_amn.run(args)

    if args.make_amn_cam_pass is True:
        import step.make_amn_cam

        timer = pyutils.Timer('step.make_amn_cam:')
        step.make_amn_cam.run(args)

    if args.eval_amn_cam_pass is True:
        import step.eval_amn_cam

        timer = pyutils.Timer('step.eval_amn_cam:')
        final_miou = []
        for i in range(65, 70):
            t = i / 100.0
            args.cam_eval_thres = t
            miou = step.eval_amn_cam.run(args)
            final_miou.append(miou)
        # print(args.cam_out_dir)
        # print(final_miou)
        print(np.max(np.array(final_miou)))

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.cam_to_mask_pass is True:
        import step.cam_to_label

        timer = pyutils.Timer('step.cam_to_label:')
        step.cam_to_label.cam_to_label(args)

    if args.eval_mask_pass is True:
        import step.eval_mask

        timer = pyutils.Timer('step.eval_mask:')
        step.eval_mask.run(args)

    if args.SAM_label_pass is True:
        import step.SAM_label, step.SAM_label_v1
        if args.SAMlabel.split('/')[-1] == 'SAMlabel':
            timer = pyutils.Timer('step.sam_to_label:')
            step.SAM_label.SAM(args)
        elif args.SAMlabel.split('/')[-1].contains('SAMlabelV1'):
            timer = pyutils.Timer('step.sam_to_label_v1:')
            step.SAM_label_v1.SAM(args)

    if args.eval_sam_pass is True:
        import step.eval_sam

        timer = pyutils.Timer('step.eval_sam:')
        step.eval_sam.run(args)

    if args.amn_cam_to_ir_label_pass is True:
        import step.amn_cam_to_ir_label

        timer = pyutils.Timer('amn_cam_to_ir_label:')
        step.amn_cam_to_ir_label.run(args)