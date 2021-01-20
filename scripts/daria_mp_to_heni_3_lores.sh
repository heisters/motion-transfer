source=data/source_daria_mp_lores
target=data/target_heni_3_lores
./build_dataset.py --dataroot $source -i ~/media/daria_movement_portrait.MTS --resize 512x288 --exclude-landmark jaw --trim 20:395 && \
./build_dataset.py --dataroot $target -i ~/media/target_heni_3.MP4 --resize 512x288 --exclude-landmark jaw && \
./train.py --dataroot $target --name target_heni_3_lores_256p @training.opt --netG global --fineSize 256 --ngf 64 --resize_or_crop scale_with_and_crop --niter 7 --niter_decay 7 && \
./train.py --dataroot $target --name target_heni_3_lores_512p @training.opt --netG local --fineSize 512 --ngf 32 --resize_or_crop none --niter 10 --niter_decay 10 --niter_fix_global 4 --load_pretrain checkpoints/target_heni_3_lores_256p && \
./generate_video.py --dataroot $source --name target_heni_3_lores_512p @testing.opt --fineSize 512 --ngf 32 --netG local --resize_or_crop none --codec prores
