name=heni_portrait
./build_dataset.py --dataroot data/$name -i ~/media/PORTRAIT_C0021.mov --resize 1024x576 --exclude-landmark jaw && \
#./build_dataset.py --dataroot data/Portrait_Ian_Puppet -i ~/media/Portrait_Ian_Puppet.mov --resize 1024x576 --flip v --exclude-landmark jaw && \
./train.py --dataroot data/$name --name "${name}_512p" @training.opt @global.opt --niter 7 --niter_decay 7 && \
./train.py --dataroot data/$name --name "${name}_1024p" @training.opt @local.opt --niter 10 --niter_decay 10 --niter_fix_global 4 --load_pretrain "checkpoints/${name}_512p"
#./generate_video.py --dataroot data/Portrait_Ian_Puppet --name Portrait-C0025_1_1024p @testing.opt @local.opt --codec prores
