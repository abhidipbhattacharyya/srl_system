### TODO: How to use predicate? In feature_extraction.process.create_batch_data
## pred is not pushed to batched data. done

# dsentences are sorted as per len. don't do it for pred. need chnges in batching
## ELMO feature. Or LM feature

##config py - done
## character CNN/RNN- read from elmo -1 and gung thesis
## Ask John for list of LSTM issue


## GAN
## create the stat class that will have normalised proba
## of having word-tag prob. P(tag|word). Using this generate a tag sequence.
## this will be used as noise in GAN

## make a fixed length for padding. So the seq length for each batch is same
## check if a seq is gt than max len. But how to handle dev/test
# case. Check he.et.al implementation. For dev/test case we need to tag
# all the sentence to be able to run eval.scripts
# size of tag input and output is different.
#Input = act_tag+2 (SOS,EOS)
#output =act_tag+1 (confused) need to understand CRF

## train a base model. Take the crf and initialise the gan crfself. done
## generate the noise from the base model. done
## take hard negative sampling from vse++ paper: https://github.com/fartashf/vsepp


## python@chenhao_GPU:
#source activate abhipython
#watch -n 2 nvidia-smi
# python predict.py /data/abhidip/srl/conll05.devel.txt /data/abhidip/srl/conll05.devel.props.gold.txt  trained_model/model.epoch400
