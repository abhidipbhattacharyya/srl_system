# may need to be put in a separate package
import os
from configuration.config import *
from util.conll_utils import *
import subprocess
#from measurements import Timer


def evaluate(pred_labels, gold_path):
    for i in range(len(pred_labels)):
        pred_labels[i] = bio_to_se(pred_labels[i])

    directory = OUTPUT_PATH
    if not os.path.exists(directory):
        os.makedirs(directory)
    temp_output = os.path.join(directory, "srl_pred_%d.tmp" % os.getpid())
    print("Printing results to temp file: {}".format(temp_output))
    print_to_conll(pred_labels, gold_path, temp_output)
    eval_script = SRL_CONLL_EVAL_SCRIPT
    child = subprocess.Popen('sh {} {} {}'.format(eval_script, gold_path, temp_output),shell = True, stdout=subprocess.PIPE)
    (eval_info, error) = child.communicate()
    eval_info = str(eval_info.decode('UTF-8'))
    #print('op=={} '.format(eval_info))

    try:
        Fscore = eval_info.strip().split("\n")[6]
        Fscore = Fscore.strip().split()[6]
        accuracy = float(Fscore)
        print(eval_info)
        print("Fscore={}".format(accuracy))
    except IndexError:
        print("Unable to get FScore. Skipping.")
