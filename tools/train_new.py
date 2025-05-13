import os
import argparse
from unittest import skip
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys

cudnn.benchmark = True
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict




def main(cfg, resume, opts):
    if cfg.DATASET.TYPE != "imagenet":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)


    if cfg.DATASET.TYPE == "imagenet":
        model_dict_type = imagenet_model_dict
    else:
        model_dict_type = cifar_model_dict

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        print('---------------vanilla train------------------')
        if cfg.DATASET.TYPE == "imagenet":
            model_student = model_dict_type[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = model_dict_type[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)

    elif cfg.DISTILLER.TYPE == "transfer":
        print('---------------transfer------------------')
        model = model_dict_type[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        weights = load_checkpoint(cfg.CAT_KD.teacher_dir)['model']
        weights_dict={}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_k = new_k.replace('student.', '') if 'student' in new_k else new_k
            weights_dict[new_k] = v
        
        weights = model.state_dict()
        
        mid = {k: v for k, v in weights_dict.items() if 'fc' not in k and 'linear' not in k}
        weights.update(mid)

        model.load_state_dict(weights)
        model_student = model
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, cfg)
    
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            # model_teacher = model_dict_type[cfg.DISTILLER.TEACHER](pretrained=True)  #imagenet有预训练权重，所以有pretrained
            # model_student = model_dict_type[cfg.DISTILLER.STUDENT](pretrained=False)
            if isinstance(model_dict_type[cfg.DISTILLER.TEACHER], dict):
                # 使用自定义预训练权重
                model_teacher = model_dict_type[cfg.DISTILLER.TEACHER][0](pretrained=False)
                pretrain_path = model_dict_type[cfg.DISTILLER.TEACHER][1]
                checkpoint = torch.load(pretrain_path)
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                # 处理权重键名
                weights_dict = {}
                for k, v in checkpoint.items():
                    new_k = k.replace('module.', '') if 'module' in k else k
                    weights_dict[new_k] = v
                model_teacher.load_state_dict(weights_dict)
                print(log_msg("Successfully loaded custom pretrained model", "INFO"))
            else:
                # 使用默认预训练权重
                model_teacher = model_dict_type[cfg.DISTILLER.TEACHER](pretrained=True)
                print(log_msg("Using default ImageNet pretrained weights", "INFO"))
            model_student = model_dict_type[cfg.DISTILLER.STUDENT](pretrained=False)
        else:  #cifar没有预训练权重，所以没有pretrained
            if cfg.CAT_KD.teacher_dir is not None:
                print('---------------custom teacher------------------')
                net = model_dict_type[cfg.DISTILLER.TEACHER][0]
                pretrain_model_path = cfg.CAT_KD.teacher_dir
                print('get teacher dir from cfg')
            else:
                print('---------------default teacher------------------')
                net, pretrain_model_path = model_dict_type[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            temp = load_checkpoint(pretrain_model_path)["model"] #加载模型
            
            weights_dict = {}
            for k, v in temp.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
            temp = weights_dict
            
            temp = {k: v for k, v in temp.items() if k in model_teacher.state_dict()}

            
            model_teacher.load_state_dict(temp)
            model_student = model_dict_type[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
            #simkd
        # if cfg.DISTILLER.TYPE == "SIMKD" or cfg.DISTILLER.TYPE == "SIMKD_test" or cfg.DISTILLER.TYPE == "SIMKD_test_2":
        if "SIMKD" in cfg.DISTILLER.TYPE or "SFAKD" in cfg.DISTILLER.TYPE:
            student_s_n = model_student.get_stage_channels()[-1]
            teacher_t_n = model_teacher.get_stage_channels()[-1]
            if cfg.DATASET.TYPE == "imagenet":
                if cfg.DISTILLER.TEACHER[0] == 'R':
                    teacher_t_n = teacher_t_n // model_teacher.get_block_expansion()
                if cfg.DISTILLER.STUDENT[0] == 'R':
                    student_s_n = student_s_n // model_student.get_block_expansion()
                print('student_s_n:', student_s_n)
                print('teacher_s_n:', teacher_t_n)
            # print(cfg.DISTILLER.TEACHER[0])
            if cfg.DISTILLER.TEACHER[0] == 'r' or cfg.DISTILLER.TEACHER[0] == 'R' or cfg.DISTILLER.TEACHER[0] == 'w':
                teacher_cls = model_teacher.fc
            elif cfg.DISTILLER.TEACHER[0] == 'v':
                teacher_cls = model_teacher.classifier
            else:
                teacher_cls = model_teacher.linear
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, student_s_n, teacher_t_n, teacher_cls
            )
            
    # convert teacher's structure
        # if cfg.if_test == True:
            
        #     if cfg.DISTILLER.TEACHER[-1] != 't':
        #         print('converting teacher\'s structure')
        #         if cfg.DATASET.TYPE == "imagenet":
        #             tnet_test = model_dict_type[cfg.DISTILLER.TEACHER+'_test'](pretrained=False)
        #         else:
        #             tnet_test = model_dict_type[cfg.DISTILLER.TEACHER+'_test'][0](num_classes=num_classes)
        #         model_dict = tnet_test.state_dict()

        #         pretrained_dict = {k: v for k, v in model_teacher.state_dict().items() if k in model_dict}                
        #         if cfg.DATASET.TYPE == "imagenet":
        #             pretrained_dict['conv_test.weight']=model_teacher.state_dict()['fc.weight'].view(model_teacher.fc.out_features,model_teacher.fc.in_features,1,1)
        #         elif cfg.DISTILLER.TEACHER == 'ResNet50' or cfg.DISTILLER.TEACHER == 'WideResNet28x10_cifar100':
        #             pretrained_dict['conv_test.weight']=model_teacher.state_dict()['linear.weight'].view(model_teacher.linear.out_features,model_teacher.linear.in_features,1,1)
        #         elif cfg.DISTILLER.TEACHER == 'vgg13':
        #             pretrained_dict['conv_test.weight']=model_teacher.state_dict()['classifier.weight'].view(model_teacher.classifier.out_features,model_teacher.classifier.in_features,1,1)
        #         else:
        #             pretrained_dict['conv_test.weight']=model_teacher.state_dict()['fc.weight'].view(model_teacher.fc.out_features,model_teacher.fc.in_features,1,1)                
        #         model_dict.update(pretrained_dict) 
        #         tnet_test.load_state_dict(model_dict)
        #         model_teacher=tnet_test
                
        #     if cfg.DATASET.TYPE == "imagenet":
        #         model_student = model_dict_type[cfg.DISTILLER.STUDENT+'_test'](pretrained=False)
        #     else:
        #         model_student = model_dict_type[cfg.DISTILLER.STUDENT+'_test'][0](num_classes=num_classes)
                
                
        elif cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())
    # dist.init_process_group(backend="nccl")
    # distiller = DDP(
    #     distiller.cuda(), device_ids=[cfg.device], find_unused_parameters=True
    # )

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    torch.cuda.empty_cache()

    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    main(cfg, args.resume, args.opts)
