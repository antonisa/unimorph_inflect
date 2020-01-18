# -*- coding: utf-8 -*-

import dynet as dy
import numpy as np
from operator import itemgetter
import os, sys
from random import random,shuffle
import re


from unimorph_inflect.utils.resources import download, languages, DEFAULT_MODEL_DIR
from .myutil import argmax, read_vocab

def change_pred_len_per_language(L2):
    MAX_PREDICTION_LEN_DEF = 20
    if L2 == "kabardian":
        MAX_PREDICTION_LEN_DEF = 25
    elif L2 == "tatar":
        MAX_PREDICTION_LEN_DEF = 23
    elif L2 == "greek":
        MAX_PREDICTION_LEN_DEF = 30
    elif L2 == "latin":
        MAX_PREDICTION_LEN_DEF = 25
    elif L2 == "livonian":
        MAX_PREDICTION_LEN_DEF = 22
    elif L2 == "bengali":
        MAX_PREDICTION_LEN_DEF = 23
    elif L2 == "czech":
        MAX_PREDICTION_LEN_DEF = 30
    elif L2 == "lithuanian":
        MAX_PREDICTION_LEN_DEF = 33
    elif L2 == "russian":
        MAX_PREDICTION_LEN_DEF = 50
    elif L2 == "irish":
        MAX_PREDICTION_LEN_DEF = 37
    elif L2 == "quechua":
        MAX_PREDICTION_LEN_DEF = 30
    elif L2 == "azeri":
        MAX_PREDICTION_LEN_DEF = 22
    elif L2 == "yiddish":
        MAX_PREDICTION_LEN_DEF = 22
    return MAX_PREDICTION_LEN_DEF

EOS = "<EOS>"
NULL = "<NULL>"

def run_lstm(init_state, input_vecs):
    s = init_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


class InflectionModel:
    def __init__(self, characters_vocab, tag_vocab, LSTM_NUM_OF_LAYERS=1, EMBEDDINGS_SIZE=32, STATE_SIZE=100, ATTENTION_SIZE=100, MINIBATCH_SIZE=1, COPY_WEIGHT=0.8, DROPOUT_PROB=0.2, EOS ="<EOS>", NULL="<NULL>", MAX_PREDICTION_LEN_DEF=20, LENGTH_NORM_WEIGHT=0.1, USE_ATT_REG=False, USE_TAG_ATT_REG=False, PREDICT_LANG=False):
        self.model = dy.Model()

        self.characters = characters_vocab
        self.tags = tag_vocab
        self.int2char = list(self.characters)
        self.char2int = {c:i for i,c in enumerate(self.characters)}

        self.int2tag = list(self.tags)
        self.tag2int = {c:i for i,c in enumerate(self.tags)}

        self.VOCAB_SIZE = len(self.characters)
        self.TAG_VOCAB_SIZE = len(self.tags)

        self.LSTM_NUM_OF_LAYERS = LSTM_NUM_OF_LAYERS
        self.EMBEDDINGS_SIZE = EMBEDDINGS_SIZE
        self.STATE_SIZE = STATE_SIZE
        self.ATTENTION_SIZE = ATTENTION_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.COPY_WEIGHT = COPY_WEIGHT
        self.DROPOUT_PROB = DROPOUT_PROB
        self.MAX_PREDICTION_LEN_DEF = MAX_PREDICTION_LEN_DEF
        self.LENGTH_NORM_WEIGHT = LENGTH_NORM_WEIGHT
        self.USE_ATT_REG=USE_ATT_REG
        self.USE_TAG_ATT_REG=USE_TAG_ATT_REG
        self.PREDICT_LANG = PREDICT_LANG

        self.EOS = EOS 
        self.NULL = NULL

        self.enc_fwd_lstm = dy.CoupledLSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.enc_bwd_lstm = dy.CoupledLSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)

        self.dec_lstm = dy.CoupledLSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.STATE_SIZE*3+self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)

        self.input_lookup = self.model.add_lookup_parameters((self.VOCAB_SIZE, self.EMBEDDINGS_SIZE) )
        self.tag_input_lookup = self.model.add_lookup_parameters((self.TAG_VOCAB_SIZE, self.EMBEDDINGS_SIZE) )
        self.attention_w1 = self.model.add_parameters( (self.ATTENTION_SIZE, self.STATE_SIZE*2) )
        self.attention_w2 = self.model.add_parameters( (self.ATTENTION_SIZE, self.STATE_SIZE*self.LSTM_NUM_OF_LAYERS*2) )
        self.attention_w3 = self.model.add_parameters( (self.ATTENTION_SIZE, 5) )
        self.attention_v = self.model.add_parameters( (1, self.ATTENTION_SIZE))

        self.decoder_w = self.model.add_parameters( (self.VOCAB_SIZE, self.STATE_SIZE))
        self.decoder_b = self.model.add_parameters( (self.VOCAB_SIZE))
        #output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
        self.output_lookup = self.input_lookup

        self.enc_tag_lstm = dy.CoupledLSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.tag_attention_w1 = self.model.add_parameters( (self.ATTENTION_SIZE, self.STATE_SIZE))
        self.tag_attention_w2 = self.model.add_parameters( (self.ATTENTION_SIZE, self.STATE_SIZE*self.LSTM_NUM_OF_LAYERS*2))
        self.tag_attention_v = self.model.add_parameters( (1, self.ATTENTION_SIZE))

        #if PREDICT_LANG:
        #    self.lang_class_w = self.model.add_parameters((STATE_SIZE*2, NUM_LANG))
        #    #self.lang_class_w = self.model.add_parameters((STATE_SIZE*2, 1))

    def embed_tags(self, tags):
        tags = [self.tag2int[t] for t in tags]
        return [self.tag_input_lookup[tag] for tag in tags]

    def embed_sentence(self, sentence):
        sentence = [self.EOS] + list(sentence) + [self.EOS]
        sentence = [self.char2int[c] for c in sentence]
        return [self.input_lookup[char] for char in sentence]


    def self_encode_tags(self, tags):
        vectors = tags
        # Self attention for every tag:
        vectors = run_lstm(self.enc_tag_lstm.initial_state(), tags)
        tag_input_mat = dy.concatenate_cols(vectors)
        out_vectors = []
        for v1 in vectors:
            # tag input mat: [tag_emb x seqlen]
            # v1: [tag_emb]
            unnormalized = dy.transpose(dy.transpose(v1) * tag_input_mat)
            self_att_weights = dy.softmax(unnormalized)
            to_add = tag_input_mat*self_att_weights
            out_vectors.append(v1 + tag_input_mat*self_att_weights)
        return out_vectors


    def encode_tags(self, tags):
        vectors = run_lstm(self.enc_tag_lstm.initial_state(), tags)
        return vectors

    def encode_sentence(self, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors

    def attend_tags(self, state, w1dt):

        w2dt = self.tag_attention_w2*state
        unnormalized = dy.transpose(self.tag_attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)        
        return att_weights

    def attend(self, state, w1dt):
        w2dt = self.attention_w2*state
        unnormalized = dy.transpose(self.attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        return att_weights

    def attend_with_prev(self, state, w1dt, prev_att):
        w2dt = self.attention_w2 * state
        w3dt = self.attention_w3 * prev_att
        unnormalized = dy.transpose(self.attention_v * dy.tanh(dy.colwise_add(dy.colwise_add(w1dt, w2dt), w3dt)))
        att_weights = dy.softmax(unnormalized)
        return att_weights

    def decode(self, vectors, tag_vectors, output, lang_id, weight, teacher_prob=1.0):
        output = [self.EOS] + list(output) + [self.EOS]
        output = [self.char2int[c] for c in output]

        N = len(vectors)

        input_mat = dy.concatenate_cols(vectors)
        w1dt = None
        input_mat = dy.dropout(input_mat, self.DROPOUT_PROB)

        tag_input_mat = dy.concatenate_cols(tag_vectors)
        tag_w1dt = None

        last_output_embeddings = self.output_lookup[self.char2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([vectors[-1], tag_vectors[-1], last_output_embeddings]))
        loss = []
        prev_att = dy.zeros(5)

        if self.USE_ATT_REG:
            total_att = dy.zeros(N)
        if self.USE_TAG_ATT_REG:
            total_tag_att = dy.zeros(len(tag_vectors))

        for char in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1 * input_mat
            tag_w1dt = tag_w1dt or self.tag_attention_w1 * tag_input_mat

            state = dy.concatenate(list(s.s()))
            
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = tag_input_mat * tag_att_weights

            tag_context2 = dy.concatenate([tag_context,tag_context])

            new_state = state + tag_context2

            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)
            context = input_mat * att_weights
            best_ic = np.argmax(att_weights.vec_value())
            context = input_mat * att_weights
            startt = min(best_ic-2,N-6)
            if startt < 0:
                startt = 0
            endd = startt+5

            if N < 5:
                prev_att = dy.concatenate([att_weights] + [dy.zeros(1)]*(5-N) )
            else:
                prev_att = att_weights[startt:endd]
            #if prev_att.dim()[0][0] != 5:
            #   print(prev_att.dim())

            if self.USE_ATT_REG:
                total_att = total_att + att_weights
            if self.USE_TAG_ATT_REG:
                total_tag_att = total_tag_att + tag_att_weights

            vector = dy.concatenate([context, tag_context, last_output_embeddings])
            s = s.add_input(vector)

            s_out = dy.dropout(s.output(), self.DROPOUT_PROB)

            out_vector = self.decoder_w * s_out + self.decoder_b
            probs = dy.softmax(out_vector)
            if teacher_prob == 1:
                last_output_embeddings = self.output_lookup[char]
            else:
                if random() > teacher_prob:
                    out_char = np.argmax(probs.npvalue())
                    last_output_embeddings = self.output_lookup[out_char]
                else:
                    last_output_embeddings = self.output_lookup[char]


            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)*weight

        
        if self.PREDICT_LANG:
            last_enc_state = vectors[-1]
            adv_state = dy.flip_gradient(last_enc_state)
            pred_lang = dy.transpose(dy.transpose(adv_state)*self.lang_class_w)
            lang_probs = dy.softmax(pred_lang)
            lang_loss_1 = -dy.log(dy.pick(lang_probs, lang_id))

            first_enc_state = vectors[0]
            adv_state2 = dy.flip_gradient(first_enc_state)
            pred_lang2 = dy.transpose(dy.transpose(adv_state2)*self.lang_class_w)
            lang_probs2 = dy.softmax(pred_lang2)
            lang_loss_2 = -dy.log(dy.pick(lang_probs2, lang_id))
            loss += lang_loss_1 + lang_loss_2

        if self.USE_ATT_REG:
            loss += dy.huber_distance(dy.ones(N),total_att)
        if self.USE_TAG_ATT_REG:
            loss += dy.huber_distance(dy.ones(len(tag_vectors)), total_tag_att)

        return loss

    def generate_nbest(self, in_seq, tag_seq, beam_size=4, show_att=False, show_tag_att=False, fn=None):
        dy.renew_cg()
        try:
            embedded = self.embed_sentence(in_seq)
        except KeyError as e:
            print(f"The character {e.args[0]} is not in the languages alphabet (which means it doesn't appear in the Unimorph data).\nIf the character is uppercase, maybe lowercasing the input could help?")
            return []

        encoded = self.encode_sentence(embedded)
        
        try:
            embedded_tags = self.embed_tags(tag_seq)
        except KeyError as e:
            print(f"The tag {e.args[0]} is not in the supported tagset for this language (which means it doesn't appear in the Unimorph data).")
            return []

        #encoded_tags = self.encode_tags(embedded_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)
        

        input_mat = dy.concatenate_cols(encoded)
        tag_input_mat = dy.concatenate_cols(encoded_tags)
        prev_att = dy.zeros(5)

        tmpinseq = [self.EOS] + list(in_seq) + [self.EOS]
        N = len(tmpinseq)

        last_output_embeddings = self.output_lookup[self.char2int[self.EOS]]
        init_vector = dy.concatenate([encoded[-1], encoded_tags[-1], last_output_embeddings])
        s_0 = self.dec_lstm.initial_state().add_input(init_vector)
        w1dt = self.attention_w1 * input_mat
        tag_w1dt = self.tag_attention_w1 * tag_input_mat

        beam = {0: [(0, s_0.s(), [], prev_att)]}

        i = 1

        nbest = []
         # we'll need this
        last_states = {}

        MAX_PREDICTION_LEN = max(len(in_seq)*1.5,self.MAX_PREDICTION_LEN_DEF)

        # expand another step if didn't reach max length and there's still beams to expand
        while i < MAX_PREDICTION_LEN and len(beam[i - 1]) > 0:
            # create all expansions from the previous beam:
            next_beam_id = []
            for hyp_id, hypothesis in enumerate(beam[i - 1]):
                # expand hypothesis tuple
                #prefix_seq, prefix_prob, prefix_decoder, prefix_context, prefix_tag_context = hypothesis
                prefix_prob, prefix_decoder, prefix_seq, prefix_att = hypothesis
                
                if i > 1:
                    last_hypo_symbol = prefix_seq[-1]
                else:
                    last_hypo_symbol = self.EOS

                # cant expand finished sequences
                if last_hypo_symbol == self.EOS and i > 3:
                    continue
                # expand from the last symbol of the hypothesis
                last_output_embeddings = self.output_lookup[self.char2int[last_hypo_symbol]]

                # Perform the forward step on the decoder
                # First, set the decoder's parameters to what they were in the previous step
                if (i == 1):
                    s = self.dec_lstm.initial_state().add_input(init_vector)
                else:
                    s = self.dec_lstm.initial_state(prefix_decoder)

                state = dy.concatenate(list(s.s()))
                tag_att_weights = self.attend_tags(state, tag_w1dt)
                tag_context = tag_input_mat * tag_att_weights
                tag_context2 = dy.concatenate([tag_context,tag_context])
                new_state = state + tag_context2

                att_weights = self.attend_with_prev(new_state, w1dt, prefix_att)
                best_ic = np.argmax(att_weights.vec_value())
                startt = min(best_ic-2, N-6)
                if startt < 0:
                    startt = 0
                endd = startt+5
                if N < 5:
                    prev_att = dy.concatenate([att_weights] + [dy.zeros(1)]*(5-N) )
                else:
                    prev_att = att_weights[startt:endd]
                #if prev_att.dim()[0][0] != 5:
                #    print(prev_att.dim())
                context = input_mat * att_weights

                vector = dy.concatenate([context, tag_context, last_output_embeddings])
                s_0 = s.add_input(vector)
                out_vector = self.decoder_w * s_0.output() + self.decoder_b
                probs = dy.softmax(out_vector).npvalue()

                # Add length norm
                length_norm = np.power(5 + i, self.LENGTH_NORM_WEIGHT)/(np.power(6,self.LENGTH_NORM_WEIGHT))
                probs = probs/length_norm


                last_states[hyp_id] = s_0.s()

                # find best candidate outputs
                n_best_indices = argmax(probs, beam_size)
                for index in n_best_indices:
                    this_score = prefix_prob + np.log(probs[index])
                    next_beam_id.append((this_score, hyp_id, index, prev_att))
                next_beam_id.sort(key=itemgetter(0), reverse=True)
                next_beam_id = next_beam_id[:beam_size]

            # Create the most probable hypotheses
            # add the most probable expansions from all hypotheses to the beam
            new_hypos = []
            for item in next_beam_id:
                hypid = item[1]
                this_prob = item[0]
                char_id = item[2]
                next_sentence = beam[i - 1][hypid][2] + [self.int2char[char_id]]
                new_hyp = (this_prob, last_states[hypid], next_sentence, item[3])
                new_hypos.append(new_hyp)
                if next_sentence[-1] == self.EOS or i == MAX_PREDICTION_LEN-1:
                    if ''.join(next_sentence) != self.EOS and ''.join(next_sentence) != self.EOS+self.EOS and ''.join(next_sentence) != self.EOS+self.EOS+self.EOS:
                        nbest.append(new_hyp)

            beam[i] = new_hypos
            i += 1
            if len(nbest) > 0:
                nbest.sort(key=itemgetter(0), reverse=True)
                nbest = nbest[:beam_size]
            if len(nbest) == beam_size and (len(new_hypos) == 0 or (nbest[-1][0] >= new_hypos[0][0])):
                break

        return nbest


    def get_loss(self, input_sentence, input_tags, output_sentence, lang_id, weight=1, tf_prob=1.0):
        embedded = self.embed_sentence(input_sentence)
        encoded = self.encode_sentence(embedded)
        embedded_tags = self.embed_tags(input_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)
        return self.decode(encoded, encoded_tags, output_sentence, lang_id, weight, tf_prob)



def test(inf_model, lemmas, tags, beam_size=4):
    outputs = []
    for lemma, tag in zip(lemmas,tags):
        lemma = list(lemma)
        tag = re.split('\W+',tag.strip())
        out = inf_model.generate_nbest(lemma, tag, beam_size)
        if len(out):
            word = ''.join([c for c in out[0][2] if c != EOS])
        elif out:
            word = ''.join([c for c in out[0][2] if c != EOS])
        else:
            word = ''.join(lemma)
        outputs.append(word)
    return outputs


def inflect(lemmas, tags, language='ell', models_dir=DEFAULT_MODEL_DIR, force_download=False, verbose=False):
    # check for models
    if verbose:
        print(f'checking for models for language {language}...')
    if language not in languages:
        raise Exception(f'language {language} not supported, sorry!')
        return -1

    lang_models_dir = '%s/%s' % (models_dir, language)
    if not os.path.exists(lang_models_dir):
        print('could not find: '+lang_models_dir)
        download(language, resource_dir=models_dir, force=force_download)

    MODEL_DIR = lang_models_dir
    character_set = read_vocab(os.path.join(lang_models_dir, "char.vocab"))
    tag_set = read_vocab(os.path.join(lang_models_dir, "tag.vocab"))

    if type(lemmas) is str and type(tags) is str:
        lemmas = [lemmas]
        tags = [tags]
    elif type(lemmas) is list and type(tags) is list:
        assert(len(lemmas) == len(tags))
    else:
        raise Exception("Error: lemmas and tags don't have the same length!")

    inflection_model = InflectionModel(character_set, tag_set, MAX_PREDICTION_LEN_DEF=change_pred_len_per_language(language))

    inflection_model.model.populate(os.path.join(lang_models_dir, language+".model"))
    results = test(inflection_model, lemmas, tags, 8)    

    return results



