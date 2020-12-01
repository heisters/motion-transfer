 ```
 ./run.py -d take_3_60s -s ~/media/take_3_nfp_full.mov --source-from 600 --source-to 660 -t ~/media/take_3_nfp_full.mov --target-from 1200 --target-to 1260
 python train.py --name take_3_60s --netG local --ngf 32 --num_D 3 --niter_fix_global 20 --label_nc 25 --resize_or_crop scale_width_and_crop --fineSize 1024 --dataroot ./data/take_3_60s/ --no_instance --fp16
 python test.py --name take_3_60s --netG local --ngf 32 --label_nc 25 --resize_or_crop none --fp16  --dataroot ./data/take_3_60s/ --no_instance
 ffmpeg -r 24 -f image2 -i results/take_3_60s/test_latest/images/%05d_synthesized_image.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
 ```
