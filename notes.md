```
./build_dataset.py --dataroot data/take_3_1m -i ~/media/take_3_nfp_full.mov --trim 600:660 --resize 576x1024
./build_dataset.py --dataroot data/take_3_8m4 -i ~/media/take_3_nfp_full.mov --trim 660:1440 --subsample 4 --resize 576x1024
./train.py --dataroot data/take_3_8m4 --name take_3_8m4_512p @training.opt --netG global --fineSize 512 --ngf 64 --niter 3 --niter_decay 3 --resize_or_crop scale_width_and_crop
./train.py --dataroot data/take_3_8m4 --name take_3_8m4_1024p @training.opt --netG local --fineSize 1024 --ngf 32 --niter 5 --niter_decay 5 --niter_fix_global 2 --resize_or_crop none --load_pretrain checkpoints/take_3_8m4_512p
./generate_video --dataroot data/take_3_1m --name take_3_8m4_1024p @testing.opt --netG local --ngf 32 --resize_or_crop none
```

```
./build_dataset.py --dataroot data/Portrait-C0025_1_b -i ~/media/Portrait-C0025_1.mp4 --resize 1024x576 --trim 0:171 --exclude-landmark jaw
./build_dataset.py --dataroot data/Portrait_Ian_Puppet_b -i ~/media/Portrait_Ian_Puppet.mov --resize 1024x576 --flip v --exclude-landmark jaw
./train.py --dataroot data/Portrait-C0025_1_b --name Portrait-C0025_1_b_512p @training.opt @global.opt --niter 3 --niter_decay 3 && \
./train.py --dataroot data/Portrait-C0025_1_b --name Portrait-C0025_1_b_1024p @training.opt @local.opt --niter 5 --niter_decay 5 --niter_fix_global 2 --load_pretrain checkpoints/Portrait-C0025_1_b_512p
./generate_video --dataroot data/Portrait_Ian_Puppet_b --name Portrait-C0025_1_b_1024p @testing.opt @local.opt
```

```
./build_dataset.py --dataroot data/Gesture_5 -i ~/media/C0005.mov --resize 288x512 --no-label --subsample 2 --train-a
./build_dataset.py --dataroot data/Gesture_5 -i ~/media/C0005.mov --resize 288x512 --no-label --subsample 2 --subsample-offset 1 --test-a
./build_dataset.py --dataroot data/Gesture_5 -i ~/media/C0003.mov --resize 288x512 --no-label --trim 0:320.32 --subsample 2 --train-b
./train.py --dataroot data/Gesture_5 --name Gesture_5_256p --num_D=3 --label_nc=0 --no_instance --fp16 --netG global --fineSize 256 --ngf 64 --niter 5 --niter_decay 5 --resize_or_crop scale_width_and_crop && \
./train.py --dataroot data/Gesture_5 --name Gesture_5_512p --num_D=3 --label_nc=0 --no_instance --fp16 --netG local --fineSize 512 --ngf 32 --niter 10 --niter_decay 10 --niter_fix_global 4 --resize_or_crop none --load_pretrain checkpoints/Gesture_5_256p
```
