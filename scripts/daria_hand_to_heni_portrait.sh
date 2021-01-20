name=daria_hand_to_heni_portrait
./build_dataset.py --dataroot data/$name -i ~/media/Daria_hand.MP4 --resize 1024x576 --no-label --train-a && \
#./build_dataset.py --dataroot data/$name -i ~/media/Daria_hand.MP4 --resize 1024x576 --no-label --subsample 2 --subsample-offset 1 --test-a && \
./build_dataset.py --dataroot data/$name -i ~/media/Portrait-C0025_1.mp4 --resize 1024x576 --no-label --trim 0:94.1 --train-b && \
./train.py --dataroot data/$name --name "${name}_512p" --num_D=3 --label_nc=0 --no_instance --fp16 --netG global --fineSize 512 --ngf 64 --niter 7 --niter_decay 7 --resize_or_crop scale_width_and_crop --no_flip && \
./train.py --dataroot data/$name --name "${name}_1024p" --num_D=3 --label_nc=0 --no_instance --fp16 --netG local --fineSize 1024 --ngf 32 --niter 12 --niter_decay 12 --niter_fix_global 4 --resize_or_crop none --load_pretrain checkpoints/"${name}_512p" --no_flip
#./generate_video.py --dataroot data/$name --name "${name}_1024p" --label_nc 0 --no_instance --fp16 --netG local --fineSize 1024 --ngf 32 --resize_or_crop none --codec prores
