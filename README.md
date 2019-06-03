# caption-transformer

This code implements the work A Bottom-Up and Top-Down Approach for Image Captioning using Transformer accepted at ICVGIP 2018.

### Disclaimer

This code is modified from [tensor2tensor](https://github.com/tensorflow/tensor2tensor), [show-attend-tell](https://github.com/yunjey/show-attend-and-tell) and uses features obtained from [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). Please refer to these links for further README information (for example, relating to other models and datasets included in the repo) and appropriate citations for these works.

### Requirements: software
1. python2.7
2. tensor2tensor==1.5.1 (see: [tensor2tensor](https://github.com/tensorflow/tensor2tensor))
3. tensorflow-gpu==1.4.1
4. Python packages you might not have: `Ipython`, `Matplotlib`, `scikit-image`

### Data Setup
Download the [MSCOCO dataset](http://mscoco.org/dataset/#download) into data/tmp/t2t_datagen_caption/train2014.zip, data/tmp/t2t_datagen_caption/val2014.zip, data/tmp/t2t_datagen_caption/test2014.zip, data/tmp/t2t_datagen_caption/ and captions_train-val2014.zip. Split the images according to the Karpathy split and place it into data/tmp/train2014,data/tmp/val2014 and data/tmp/test2014.  The bottom-up features are obtained from [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), for the entire coco dataset.Place it in the data/tmp/bottom_up_fatures folder.

### Data Generation

t2t-datagen --data_dir=data/t2t_data_caption_bottom --tmp_dir=data/tmp/t2t_datagen_caption --problem=image_ms_coco_tokens32k

### Training

t2t-trainer --data_dir=data/t2t_data_caption_bottom  --problems=image_ms_coco_tokens32k --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=data/t2t_train/image_ms_coco_tokens32k/transformer-transformer_base_single_gpu --keep_checkpoint_max=10 --hparams="num_heads=8" --local_eval_frequency=20000 --train_steps=550000

### Decoding 

t2t-decoder --data_dir=data/t2t_data_caption_bottom --problems=image_ms_coco_tokens32k --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=/data/t2t_train/image_ms_coco_tokens32k/transformer-transformer_base_single_gpu --decode_hparams="beam_size=1,save_images=True" --decode_to_file=dec --hparams="num_heads=8"

Evaluate generated captions with evaluate/evaluate_generated_captions.py and visualize attention with visualization/TransformerVisualizationCaption.py
