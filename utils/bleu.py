import cPickle as pickle
import os
import sys
'''
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu

from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
'''
from tensor2tensor.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from tensor2tensor.coco_caption.pycocoevalcap.rouge.rouge import Rouge
from tensor2tensor.coco_caption.pycocoevalcap.cider.cider import Cider
from tensor2tensor.coco_caption.pycocoevalcap.meteor.meteor import Meteor


def score_all(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def score_cider(ref, hypo):

    scorers = [
        #       (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        #(Meteor(), "METEOR"),
        #(Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
#        (Spice(), "SPICE")
    ]
    final_scores = {}
    final_scores_list = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
            final_scores_list[method] = scores

    return final_scores, final_scores_list

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"])
        #(Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def evaluate_for_particular_captions(cand, data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" % (split, split))


    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    # with open(candidate_path, 'rb') as f:
    #     cand = pickle.load(f)

    # make dictionary
    hypo = {}
    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    # compute bleu score
    final_scores = score_all(refe, hypo)

    # print out scores

    return final_scores


def evaluate_captions(ref,cand):
    hypo = {}

    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    final_scores = score(refe, hypo)
    #return final_scores['CIDEr']
    # return final_scores['Bleu_1']
    return 1*final_scores['Bleu_4'] + 1*final_scores['Bleu_3'] + 0.5*final_scores['Bleu_1'] + 0.5*final_scores['Bleu_2']


def evaluate_captions_bleu_4(ref, cand):
    hypo = {}

    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    final_scores = score(refe, hypo)
    # return final_scores['CIDEr']
    # return final_scores['Bleu_1']
    return final_scores['Bleu_4']


def evaluate_captions_cider(ref, cand):
    hypo = {}
    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    final_scores, final_score_list = score_cider(refe, hypo)
    return final_score_list['CIDEr']

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" %(split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" %(split, split))
    
    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)
    
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    
    # compute bleu score
    final_scores = score_all(ref, hypo)

    # print out scores
    print 'Bleu_1:\t',final_scores['Bleu_1']  
    print 'Bleu_2:\t',final_scores['Bleu_2']  
    print 'Bleu_3:\t',final_scores['Bleu_3']  
    print 'Bleu_4:\t',final_scores['Bleu_4']  
    print 'METEOR:\t',final_scores['METEOR']  
    print 'ROUGE_L:',final_scores['ROUGE_L']  
    print 'CIDEr:\t',final_scores['CIDEr']
    
    if get_scores:
        return final_scores


if __name__ == "__main__":
    ref = [[u'a tiddy bear',u'a animal'],[u'<START> a number of luggage bags on a cart in a lobby .', u'<START> a cart filled with suitcases and bags .', u'<START> trolley used for transporting personal luggage to guests rooms .', u'<START> wheeled cart with luggage at lobby of commercial business .', u'<START> a luggage cart topped with lots of luggage .']]
    dec = [u'some one',u' a man is standing next to a car with a suitcase .']
    r = [evaluate_captions([k], [v]) for k, v in zip(ref, dec)]
    print r




    
   
    
    
    
    
    
    
    
    
    
    


