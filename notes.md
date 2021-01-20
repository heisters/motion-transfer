```
./build_dataset.py --dataroot data/take_3_1m -i ~/media/take_3_nfp_full.mov --trim 600:660 --resize 576x1024
./build_dataset.py --dataroot data/take_3_8m4 -i ~/media/take_3_nfp_full.mov --trim 660:1440 --subsample 4 --resize 576x1024
./train.py --dataroot data/take_3_8m4 --name take_3_8m4_512p @training.opt --netG global --fineSize 512 --ngf 64 --niter 3 --niter_decay 3 --resize_or_crop scale_width_and_crop
./train.py --dataroot data/take_3_8m4 --name take_3_8m4_1024p @training.opt --netG local --fineSize 1024 --ngf 32 --niter 5 --niter_decay 5 --niter_fix_global 2 --resize_or_crop none --load_pretrain checkpoints/take_3_8m4_512p
./generate_video --dataroot data/take_3_1m --name take_3_8m4_1024p @testing.opt --netG local --ngf 32 --resize_or_crop none
```

```
./build_dataset.py --dataroot data/Portrait-C0025_1 -i ~/media/Portrait-C0025_1.mp4 --resize 1024x576 --trim 0:171 --exclude-landmark jaw && \
./build_dataset.py --dataroot data/Portrait_Ian_Puppet -i ~/media/Portrait_Ian_Puppet.mov --resize 1024x576 --flip v --exclude-landmark jaw && \
./train.py --dataroot data/Portrait-C0025_1 --name Portrait-C0025_1_512p @training.opt @global.opt --niter 7 --niter_decay 7 && \
./train.py --dataroot data/Portrait-C0025_1 --name Portrait-C0025_1_1024p @training.opt @local.opt --niter 10 --niter_decay 10 --niter_fix_global 4 --load_pretrain checkpoints/Portrait-C0025_1_512p && \
./generate_video.py --dataroot data/Portrait_Ian_Puppet --name Portrait-C0025_1_1024p @testing.opt @local.opt --codec prores
```

```
./build_dataset.py --dataroot data/Gesture_5 -i ~/media/C0005.mov --resize 576x1024 --no-label --subsample 2 --train-a && \
./build_dataset.py --dataroot data/Gesture_5 -i ~/media/C0005.mov --resize 576x1024 --no-label --subsample 2 --subsample-offset 1 --test-a && \
./build_dataset.py --dataroot data/Gesture_5 -i ~/media/C0003.mov --resize 576x1024 --no-label --trim 0:320.32 --subsample 2 --train-b && \
./train.py --dataroot data/Gesture_5 --name Gesture_5_512p --num_D=3 --label_nc=0 --no_instance --fp16 --netG global --fineSize 512 --ngf 64 --niter 7 --niter_decay 7 --resize_or_crop scale_width_and_crop --no_flip && \
./train.py --dataroot data/Gesture_5 --name Gesture_5_1024p --num_D=3 --label_nc=0 --no_instance --fp16 --netG local --fineSize 1024 --ngf 32 --niter 12 --niter_decay 12 --niter_fix_global 4 --resize_or_crop none --load_pretrain checkpoints/Gesture_5_512p --no_flip
./generate_video.py --dataroot data/Gesture_5 --name Gesture_5_1024p --label_nc 0 --no_instance --fp16 --netG local --fineSize 1024 --ngf 32 --resize_or_crop none --codec prores
```

```
./build_dataset.py --dataroot data/source_heni_3 -i ~/media/source_heni_3_trimmed.mp4 --resize 1024x576 --exclude-landmark jaw && \
./build_dataset.py --dataroot data/target_daria_3 -i ~/media/target_daria_3_trimmed.mov --resize 1024x576 --exclude-landmark jaw
./train.py --dataroot data/target_daria_3 --name target_daria_3_512p @training.opt @global.opt --niter 7 --niter_decay 7 \
&& ./train.py --dataroot data/target_daria_3 --name target_daria_3_1024p @training.opt @local.opt --niter 10 --niter_decay 10 --niter_fix_global 4 --load_pretrain checkpoints/target_daria_3_512p
./generate_video.py --dataroot data/source_heni_3 --name target_daria_3_1024p @testing.opt @local.opt --codec prores
```

